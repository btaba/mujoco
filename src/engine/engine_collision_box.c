// Copyright 2016 Svetoslav Kolev
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>

#include "engine/engine_collision_convex.h"
#include "engine/engine_collision_primitive.h"
#include "engine/engine_inline.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_misc.h"

// hard-clamp vector to range [-limit(i), +limit(i)]
static void mju_clampVec(mjtNum* vec, const mjtNum* limit, int n) {
  for (int i = 0; i < n; i++) {
    // loop over active limits
    if (limit[i] > 0) {
      vec[i] = mju_clip(vec[i], -limit[i], limit[i]);
    }
  }
}


// raw sphere : box
int mjraw_SphereBox(mjPreContact* con, mjtNum margin,
                    const mjtNum* pos1, const mjtNum* mat1, const mjtNum* size1,
                    const mjtNum* pos2, const mjtNum* mat2, const mjtNum* size2) {
  int i, k;
  mjtNum tmp[3], center[3], clamped[3], deepest[3];
  mjtNum pos[3];
  mjtNum dist, closest;

  mji_sub3(tmp, pos1, pos2);
  mji_mulMatTVec3(center, mat2, tmp);

  mji_copy3(clamped, center);
  mju_clampVec(clamped, size2, 3);

  mji_copy3(deepest, center);
  mji_sub3(tmp, clamped, center);
  dist = mju_normalize3(tmp);

  if (dist - size1[0] > margin)
    return 0;

  // sphere center inside box
  if (dist <= mjMINVAL) {
    closest = (size2[0] + size2[1] + size2[2]) * 2;

    for (i = 0; i < 6; i++) {
      if (closest > mju_abs((i % 2 ? 1 : -1)*size2[i / 2] - center[i / 2])) {
        closest = mju_abs((i % 2 ? 1 : -1) * size2[i / 2] - center[i / 2]);
        k = i;
      }
    }

    mjtNum nearest[3] = {0};
    nearest[k / 2] = (k % 2 ? -1 : 1);

    mji_copy3(pos, center);
    mji_addToScl3(pos, nearest, (size1[0] - closest) / 2);
    mji_mulMatVec3(con[0].normal, mat2, nearest);
    dist = -closest;
  } else {
    mji_addToScl3(deepest, tmp, size1[0]);
    mju_zero3(pos);
    mji_addToScl3(pos, clamped, 0.5);
    mji_addToScl3(pos, deepest, 0.5);
    mji_mulMatVec3(con[0].normal, mat2, tmp);
  }

  mji_mulMatVec3(tmp, mat2, pos);
  mji_add3(con[0].pos, tmp, pos2);
  con[0].dist = dist - size1[0];
  mji_zero3(con[0].tangent);
  return 1;
}


// sphere : box
int mjc_SphereBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  const mjtNum* pos1  = d->geom_xpos + 3*g1;
  const mjtNum* mat1  = d->geom_xmat + 9*g1;
  const mjtNum* size1 = m->geom_size + 3*g1;
  const mjtNum* pos2  = d->geom_xpos + 3*g2;
  const mjtNum* mat2  = d->geom_xmat + 9*g2;
  const mjtNum* size2 = m->geom_size + 3*g2;
  return mjraw_SphereBox(con, margin, pos1, mat1, size1, pos2, mat2, size2);
}


/* GENERAL THEORY OF OPERATION
   the following code is mostly for finding (line segment)/(box) collision
   after which box-sphere is called

   First the closest point to the box is found.
   Then a "sensible" second point is found if the angle
   between the segment and the box is low enough < 45

   In the comments that follow, capsule just means the capsule's line segment
   It might be hard to understand all comments but you would need
   a picture to see what is happening at each line of the code
*/

// raw capsule : box
int mjraw_CapsuleBox(mjPreContact* con, mjtNum margin,
                     const mjtNum* pos1, const mjtNum* mat1, const mjtNum* size1,
                     const mjtNum* pos2, const mjtNum* mat2,
                     const mjtNum* size2) {
  mjtNum tmp1[3], tmp2[3], tmp3[3], halfaxis[3], axis[3], dif[3];
  mjtNum pos[3];          // position of capsule in box-local frame

  mjtNum halflength;      // half of capsule's length
  mjtNum bestdist;        // closest contact point distance
  mjtNum bestdistmax;     // init value for bestdist
  mjtNum bestsegmentpos;  // between -1 and 1 :  which point on the segment is closest to the box
  mjtNum secondpos;       // distance of 2nd contact position on capsule segment from the first
  mjtNum dist;
  mjtNum bestboxpos;      // closest contact point, position on the box's edge
  mjtNum mul, e1, e2, dp, de;
  // mjtNum penetration;

  mjtNum ma, mb, mc, u, v, det, x1, x2, idet;  // linelinedist temps

  int s1, s2;        // hold linelinedist info
  int i, j, c1, c2;  // temporary variables
  int cltype = -4;   // closest type
  int clface;        // closest face
  int clcorner = 0;  // closest corner (0..7 in binary)
  int cledge;        // closest edge axis
  int axisdir;       // direction of capsule axis in relation to the box
  int n;             // number of contacts
  int ax1, ax2, ax;  // axis temporaries


  halflength = size1[1];
  secondpos = -4;  // initialize to no 2nd contact (valid values are between -1 and 1)

  mji_sub3(tmp1, pos1, pos2);       // bring capsule to box-local frame (center's box is at (0,0,0))
  mji_mulMatTVec3(pos, mat2, tmp1);  // and axis parallel to world

  tmp1[0] = mat1[2];  // capsule's axis
  tmp1[1] = mat1[5];
  tmp1[2] = mat1[8];

  mji_mulMatTVec3(axis, mat2, tmp1);      // do the same for the capsule axis
  mji_scl3(halfaxis, axis, halflength);  // scale to get actual capsule half-axis

  axisdir = 0;
  if (halfaxis[0] > 0)
    axisdir += 1;
  if (halfaxis[1] > 0)
    axisdir += 2;
  if (halfaxis[2] > 0)
    axisdir += 4;

  // under this notion    "axisdir" and "7-axisdir" point in opposite directions,
  // essentially the same for a capsule

  bestdistmax = margin + 2 * (size1[0] + halflength + size2[0] + size2[1] +
                              size2[2]);  // initialize bestdist
  bestdist = bestdistmax;
  bestsegmentpos = 0;

  mju_zero3(tmp2);

  // test to see if maybe the a face of the box is closest to the capsule
  for (i = -1; i <= 1; i += 2) {
    mji_copy3(tmp1, pos);
    mji_addToScl3(tmp1, halfaxis, i);
    mji_copy3(tmp2, tmp1);

    for (c1 = 0, j = 0, c2 = -1; j < 3; j++) {
      if (tmp1[j] < -size2[j]) {
        c1++;
        c2 = j;
        tmp1[j] = -size2[j];
      } else if (tmp1[j] > size2[j]) {
        c1++;
        c2 = j;
        tmp1[j] = size2[j];
      }
    }

    if (c1 > 1)
      continue;

    mji_subFrom3(tmp1, tmp2);
    dist = mju_dot3(tmp1, tmp1);

    if (dist < bestdist) {
      bestdist = dist;
      bestsegmentpos = i;
      cltype = -2 + i;
      clface = c2;
    }
  }

  mju_zero3(tmp2);

  for (j = 0; j < 3; j++) {
    for (i = 0; i < 8; i++) {
      if ((i & (1 << j)) == 0) {
        // trick to get a corner
        tmp3[0] = ((i & 1) ? 1 : -1) * size2[0];
        tmp3[1] = ((i & 2) ? 1 : -1) * size2[1];
        tmp3[2] = ((i & 4) ? 1 : -1) * size2[2];
        tmp3[j] = 0;

        // tmp3 is the starting point on the box
        // tmp2 is the direction along the "j"-th axis
        // pos is the capsule's center
        // halfaxis is the capsule direction

        // find closest point between capsule and the edge

        mji_sub3(dif, tmp3, pos);

        ma = size2[j] * size2[j];
        mb = -size2[j] * halfaxis[j];
        mc = size1[1] * size1[1];

        u = -size2[j] * dif[j];
        v = mju_dot3(halfaxis, dif);

        det = ma * mc - mb * mb;
        if (mju_abs(det) < mjMINVAL)
          continue;
        idet = 1 / det;


        // sX : X=1 means middle of segment. X=0 or 2 one or the other end

        x1 = (mc * u - mb * v) * idet;
        x2 = (ma * v - mb * u) * idet;

        s1 = s2 = 1;

        if (x1 > 1) {
          x1 = 1;
          s1 = 2;
          x2 = (v - mb) * (1 / mc);
        } else if (x1 < -1) {
          x1 = -1;
          s1 = 0;
          x2 = (v + mb) * (1 / mc);
        }

        if (x2 > 1) {
          x2 = 1;
          s2 = 2;
          x1 = (u - mb) * (1 / ma);
          if (x1 > 1)
            x1 = 1, s1 = 2;
          else if (x1 < -1)
            x1 = -1, s1 = 0;
        } else if (x2 < -1) {
          x2 = -1;
          s2 = 0;
          x1 = (u + mb) * (1 / ma);
          if (x1 > 1)
            x1 = 1, s1 = 2;
          else if (x1 < -1)
            x1 = -1, s1 = 0;
        }

        mji_sub3(dif, tmp3, pos);

        mji_addToScl3(dif, halfaxis, -x2);
        dif[j] += size2[j] * x1;

        tmp1[2] = mju_dot3(dif, dif);

        c1 = s1 * 3 + s2;


        // the -MINVAL might not be necessary. Fixes numerical problem when axis is numerically
        // parallel to the box
        if (tmp1[2] < bestdist - mjMINVAL) {
          bestdist = tmp1[2];
          bestsegmentpos = x2;
          bestboxpos = x1;

          // c1<6 means that closest point on the box is at the lower end
          // or in the middle of the edge
          c2 = c1 / 6;

          clcorner = i + (1 << j) * c2;  // which corner is the closest
          cledge = j;   // which axis
          cltype = c1;  // save clamped info
        }
      }
    }
  }


  // penetration = -bestdist;


  for (j = 0; j < 3; j++) {
    if (j == 2) {
      typedef union {
        struct {
          mjtNum x, y;
        };
        mjtNum c[2];
      } d2;
      d2 p, s, dd /*, c, tmp1*/;
      mjtNum uu, vv, w, ee1, best /* ,e2 */, l /* , e3, e4 */;

      bestdist = bestdistmax;

      p.x = pos[0];
      p.y = pos[1];
      dd.x = halfaxis[0];
      dd.y = halfaxis[1];
      s.x = size2[0];
      s.y = size2[1];

      l = sqrt(dd.x * dd.x + dd.y * dd.y);

      uu = dd.x * s.y;
      vv = dd.y * s.x;
      w = dd.x * p.y - dd.y * p.x;


      best = -1;

      ee1 = +uu - vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 0;
        }
      }
      ee1 = -uu - vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 1;
        }
      }
      ee1 = +uu + vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 2;
        }
      }
      ee1 = -uu + vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 3;
        }
      }

      // c.x = s.x * ((c1 / 2) ? -1 : 1);
      // c.y = s.y * ((c1 % 2) ? -1 : 1);

      ee1 = mju_abs(w) / l;
      // e2 = best / l;

      // printf("%g %g      %g %g     %g %g\n",c.x,c.y,d.x,d.y,e1,e2);

      // tmp1.x = c.x - p.x;
      // tmp1.y = c.y - p.y;
      ee1 = dd.x * dd.x + dd.y * dd.y;
      // e2 = tmp1.x * d.x + tmp1.y * d.y;
      // e3 = e2 / e1;


      // printf("%g %g      %g %g     %g %g %g \n",c.x,c.y,d.x,d.y,e1,e2,e3);

      ee1 = p.x + (+s.y - p.y) / dd.y * dd.x;
      // e2 = p.x + (-s.y - p.y) / d.y * d.x;
      // e3 = p.y + (+s.x - p.x) / d.x * d.y;
      // e4 = p.y + (-s.x - p.x) / d.x * d.y;


      // printf("%g %g     %g %g\n",e1,e2,e3,e4);
    }
  }

  // goto skip;   // allow only the closest contact

  // cltype: -3 -1 : face is closest to the capsule
  // cltype: 0..8 : edge is closest to the capsule
  // cltype/3==0 means the lower corner is closest to the capsule (note that edges include corners)
  // cltype/3==2 means the upper corner is closest to the capsule (note that edges include corners)
  // cltype/3==1 means the middle of the edge is closest to the capsule
  // cltype%3==0 means the lower corner is closest to the box (note that edges include corners)
  // cltype%3==2 means the upper corner is closest to the box (note that edges include corners)
  // cltype%3==1 means the middle of the capsule is closest to the box


  // invalid type
  if (cltype == -4)
    return 0;

  if (cltype >= 0 && cltype / 3 != 1) {  // closest to a corner of the box
    c1 = axisdir ^ clcorner;

    // hack to find the relative orientation of capsule and corner
    // there are 2 cases:
    //    1: pointing to or away from the corner
    //    2: oriented along a face or an edge


    if (c1 == 0 || c1 == 7)
      goto skip;  // case 1: no chance of additional contact

    if (c1 == 1 || c1 == 2 || c1 == 4) {
      mul = 1;
      de = 1 - bestsegmentpos;
      dp = 1 + bestsegmentpos;
    }

    if (c1 == 3 || c1 == 5 || c1 == 6) {
      mul = -1;
      c1 = 7 - c1;
      dp = 1 - bestsegmentpos;
      de = 1 + bestsegmentpos;
    }

    // "de" and "dp" distance from first closest point on the capsule to both ends of it
    // mul is a direction along the capsule's axis

    if (c1 == 1)
      ax = 0, ax1 = 1, ax2 = 2;
    if (c1 == 2)
      ax = 1, ax1 = 2, ax2 = 0;
    if (c1 == 4)
      ax = 2, ax1 = 0, ax2 = 1;



    if (axis[ax]*axis[ax] > 0.5) {  // second point along the edge of the box
      secondpos = de;  // initial position from the
      e1 = 2 * size2[ax] / mju_abs(halfaxis[ax]);

      if (e1 < secondpos) {
        secondpos = e1;  // we overshoot, move back to the  other corner of the edge
      }
      secondpos *= mul;
    } else {  // second point along a face of the box
      secondpos = dp;

      // check for overshoot again

      e1 = 2 * size2[ax1] / mju_abs(halfaxis[ax1]);
      if (e1 < secondpos)
        secondpos = e1;

      e1 = 2 * size2[ax2] / mju_abs(halfaxis[ax2]);
      if (e1 < secondpos)
        secondpos = e1;

      secondpos *= -mul;
    }
  } else if (cltype >= 0 && cltype / 3 == 1) {  // we are on box's edge
    // hacks to find the relative orientation of capsule and edge
    // there are 2 cases:
    //    c1= 2^n: edge and capsule are oriented in a T configuration (no more contacts
    //    c1!=2^n: oriented in a cross X configuration

    c1 = axisdir ^ clcorner;  // same trick

    c1 &= 7 - (1 << cledge);  // even more hacks

    // printf("%d %d %d %d    %lf %lf %lf\n",
    //        axisdir,clcorner,c1,cledge,halfaxis[0],halfaxis[1],halfaxis[2]);

    if (c1 != 1 && c1 != 2 && c1 != 4)
      goto skip;


    if (cledge == 0)
      ax1 = 1, ax2 = 2;
    if (cledge == 1)
      ax1 = 2, ax2 = 0;
    if (cledge == 2)
      ax1 = 0, ax2 = 1;
    ax = cledge;


    // Then it finds with which face the capsule has a lower angle and switches the axis names

    if (mju_abs(axis[ax1]) > mju_abs(axis[ax2]))
      ax1 = ax2;
    ax2 = 3 - ax - ax1;

    // keep track of the axis orientation (mul will tell us which direction along the capsule to
    // find the second point) you can notice all other references to the axis "halfaxis" are with
    // absolute value

    if (c1 & (1 << ax2)) {
      mul = 1;
      secondpos = 1 - bestsegmentpos;
    } else {
      mul = -1;
      secondpos = 1 + bestsegmentpos;
    }


    // now we have to find out whether we point towards the opposite side or towards one of the
    // sides and also find the farthest point along the capsule that is above the box

    e1 = 2 * size2[ax2] / mju_abs(halfaxis[ax2]);
    if (e1 < secondpos)
      secondpos = e1;

    if (((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0))  // that is insane
      e2 = 1 - bestboxpos;
    else
      e2 = 1 + bestboxpos;

    e1 = size2[ax] * e2 / mju_abs(halfaxis[ax]);

    if (e1 < secondpos)
      secondpos = e1;

    secondpos *= mul;
  } else if (cltype < 0) {
    // similarly we handle the case when one capsule's end is closest to a face of the box
    // and find where is the other end pointing to and clamping to the farthest point
    // of the capsule that's above the box

    if (clface == -1)
      goto skip;  // here the closest point is inside the box, no need for a second point
    if (cltype == -3)
      mul = 1;
    else
      mul = -1;

    secondpos = 2;

    mji_copy3(tmp1, pos);
    mji_addToScl3(tmp1, halfaxis, -mul);

    for (i = 0; i < 3; i++) {
      if (i != clface) {
        e1 = (size2[i] - tmp1[i]) / halfaxis[i] * mul;
        if (e1 > 0)
          if (e1 < secondpos)
            secondpos = e1;

        e1 = (-size2[i] - tmp1[i]) / halfaxis[i] * mul;
        if (e1 > 0)
          if (e1 < secondpos)
            secondpos = e1;
      }
    }
    secondpos *= mul;
  }


skip:

  // create sphere in original orientation at first contact point
  mju_copy3(tmp1, pos);
  mji_addToScl3(tmp1, halfaxis, bestsegmentpos);
  mju_mulMatVec3(tmp2, mat2, tmp1);
  mju_addTo3(tmp2, pos2);

  // collide with
  n = mjraw_SphereBox(con, margin, tmp2, mat1, size1, pos2, mat2, size2);


  if (secondpos > -3) {  // secondpos was modified
    mju_copy3(tmp1, pos);
    mji_addToScl3(tmp1, halfaxis, secondpos + bestsegmentpos);  // note the summation
    mju_mulMatVec3(tmp2, mat2, tmp1);
    mju_addTo3(tmp2, pos2);
    n += mjraw_SphereBox(con + n, margin, tmp2, mat1, size1, pos2, mat2, size2);
  }

  return n;
}


// capsule : box
int mjc_CapsuleBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  const mjtNum* pos1  = d->geom_xpos + 3*g1;
  const mjtNum* pos2  = d->geom_xpos + 3*g2;
  const mjtNum* mat1  = d->geom_xmat + 9*g1;
  const mjtNum* mat2  = d->geom_xmat + 9*g2;
  const mjtNum* size1 = m->geom_size + 3*g1;
  const mjtNum* size2 = m->geom_size + 3*g2;
  return mjraw_CapsuleBox(con, margin, pos1, mat1, size1, pos2, mat2, size2);
}


//------------------------------ box : box --------------------------------------------------------

// A box-box contact manifold is computed in two stages.
//
// Stage 1, separating-axis test: find the axis of maximum separation among the 15 candidate
// directions (3 face normals per box, 9 cross products of edge directions). If the boxes are
// separated by more than margin along any candidate axis there is no contact. Face axes are
// preferred over edge axes on near-ties: a face axis yields a multi-point manifold, which the
// solver strongly prefers over a single edge contact of nearly identical depth.
//
// Stage 2, manifold generation, depends on the kind of winning axis:
//  - face axis: the owner of the face is the reference box. The face of the other (incident)
//    box least aligned with the reference normal is clipped against the four side planes of
//    the reference face (Sutherland-Hodgman). Every clipped vertex within the margin band
//    becomes a contact. Depth is the distance between the surfaces along the reference
//    normal; contact position is midway between the surfaces along the normal, so it lies
//    inside the intersection of the margin-inflated boxes.
//  - edge axis: the contact is at the midpoint of the closest-point pair between the two
//    supporting edge segments, with depth measured along the separating axis.
//
// At most mjBOXBOX_MAXCON contacts are returned: the deepest vertex plus up to three more
// chosen to maximize manifold area.

// maximum number of contacts returned by box-box
#define mjBOXBOX_MAXCON 4

// relative penalty applied to edge-axis separation on near-ties with the best face axis
#define mjBOXBOX_EDGEBIAS 1e-6

// vertex capacity for face clipping: 4-gon clipped by 4 half-planes has at most 8 vertices
#define mjBOXBOX_MAXVERT 12

// clip polygon *cur (nin vertices) against the half-plane sign*v[coord] <= limit; when
// every vertex is already inside, *cur is left untouched (no copies, the common resting
// case); otherwise the result is written to spare and the buffers are swapped; vertices
// are (x, y, z) with z interpolated as an attribute; returns the vertex count
static int clipHalfPlane(int nin, mjtNum (**cur)[3], mjtNum (**spare)[3],
                         int coord, mjtNum sign, mjtNum limit) {
  mjtNum (*in)[3] = *cur;
  mjtNum d[mjBOXBOX_MAXVERT];
  int all_inside = 1;
  for (int k = 0; k < nin; k++) {
    d[k] = sign*in[k][coord] - limit;
    all_inside &= d[k] <= 0;
  }
  if (all_inside) {
    return nin;
  }

  mjtNum (*out)[3] = *spare;
  int nout = 0;
  for (int k = 0; k < nin; k++) {
    const mjtNum* p = in[k];
    int k1 = k + 1 == nin ? 0 : k + 1;
    mjtNum dp = d[k], dq = d[k1];

    // emit p if inside
    if (dp <= 0 && nout < mjBOXBOX_MAXVERT) {
      mji_copy3(out[nout++], p);
    }

    // emit intersection if the edge crosses the plane
    if ((dp < 0) != (dq < 0) && dp != dq && nout < mjBOXBOX_MAXVERT) {
      const mjtNum* q = in[k1];
      mjtNum t = dp / (dp - dq);
      out[nout][0] = p[0] + t*(q[0] - p[0]);
      out[nout][1] = p[1] + t*(q[1] - p[1]);
      out[nout][2] = p[2] + t*(q[2] - p[2]);
      nout++;
    }
  }
  *cur = out;
  *spare = in;
  return nout;
}


// from n candidate vertices, select up to mjBOXBOX_MAXCON, maximizing manifold area in the
// (x, y) plane; returns the selection count
//
// the seed is the first vertex in clip order, NOT the deepest: clip order is a deterministic
// function of the geometry, while near-equal depths flip an argmin seed between steps by
// rounding noise, churning the selected subset and destabilizing resting stacks through the
// solver's warm start; the deepest vertex is guaranteed a slot afterwards
static int reduceManifold(int n, const mjtNum (*vert)[3], int sel[mjBOXBOX_MAXCON]) {
  if (n <= mjBOXBOX_MAXCON) {
    for (int i = 0; i < n; i++) {
      sel[i] = i;
    }
    return n;
  }

  int chosen[mjBOXBOX_MAXVERT] = {0};

  // a: first vertex in clip order, for temporal stability
  int a = 0;
  chosen[a] = 1;
  sel[0] = a;

  // b: farthest from a
  int b = -1;
  mjtNum best = -1;
  for (int i = 0; i < n; i++) {
    if (chosen[i]) continue;
    mjtNum dx = vert[i][0] - vert[a][0], dy = vert[i][1] - vert[a][1];
    mjtNum d = dx*dx + dy*dy;
    if (d > best) {
      best = d;
      b = i;
    }
  }
  chosen[b] = 1;
  sel[1] = b;

  // c: maximum triangle area with (a, b)
  int c = -1;
  best = -1;
  for (int i = 0; i < n; i++) {
    if (chosen[i]) continue;
    mjtNum area = mju_abs((vert[b][0] - vert[a][0]) * (vert[i][1] - vert[a][1]) -
                          (vert[b][1] - vert[a][1]) * (vert[i][0] - vert[a][0]));
    if (area > best) {
      best = area;
      c = i;
    }
  }
  chosen[c] = 1;
  sel[2] = c;

  // d: maximum summed area against edges (a, c) and (b, c)
  int d = -1;
  best = -1;
  for (int i = 0; i < n; i++) {
    if (chosen[i]) continue;
    mjtNum area1 = mju_abs((vert[c][0] - vert[a][0]) * (vert[i][1] - vert[a][1]) -
                           (vert[c][1] - vert[a][1]) * (vert[i][0] - vert[a][0]));
    mjtNum area2 = mju_abs((vert[c][0] - vert[b][0]) * (vert[i][1] - vert[b][1]) -
                           (vert[c][1] - vert[b][1]) * (vert[i][0] - vert[b][0]));
    if (area1 + area2 > best) {
      best = area1 + area2;
      d = i;
    }
  }
  sel[3] = d;
  chosen[d] = 1;

  // the deepest vertex must be in the manifold: without it the solver under-resolves the
  // penetration; replace the last area pick if it was left out
  int deepest = 0;
  for (int i = 1; i < n; i++) {
    if (vert[i][2] < vert[deepest][2]) {
      deepest = i;
    }
  }
  if (!chosen[deepest]) {
    sel[3] = deepest;
  }
  return 4;
}



//-------------------------- legacy box-box collider ------------------------------------------
// the pre-rewrite implementation, preserved verbatim behind mjENBL_BOXBOXLEGACY for
// side-by-side comparison; see mjc_BoxBox for the dispatch

#ifdef mjUSESINGLE
  #define mjDEPTHSLACKREL 1e-4f
  #define mjDEPTHSLACKABS 1e-5f
#else
  #define mjDEPTHSLACKREL 1e-6
  #define mjDEPTHSLACKABS 1e-12
#endif

static int boxboxLegacyRaw(const mjModel* M, const mjData* D, mjPreContact* con,
                           int g1, int g2, mjtNum margin) {
  const mjtNum* pos1 = D->geom_xpos + 3 * g1;
  const mjtNum* pos2 = D->geom_xpos + 3 * g2;
  const mjtNum* mat1 = D->geom_xmat + 9 * g1;
  const mjtNum* mat2 = D->geom_xmat + 9 * g2;
  const mjtNum* size1 = M->geom_size + 3 * g1;
  const mjtNum* size2 = M->geom_size + 3 * g2;

  mjtNum pos12[3], pos21[3], rot[9], rott[9], rotabs[9], rottabs[9], tmp1[3], tmp2[3], plen1[3],
         plen2[3];
  mjtNum rotmore[9], p[3], r[9], s[3], ss[3], lp[3], rt[9], points[mjMAXCONPAIR][3],
         depth[mjMAXCONPAIR], pts[6][3], ppts2[4][2], pu[4][3], axi[3][3];
  mjtNum linesu[4][6], lines[4][6], clnorm[3], rnorm[3];
  mjtNum penetration, c1, c2, c3, a, b, c, d, lx, ly, hz, l, x, y, u, v, llx, lly, innorm, margin2;
  mjtNum maxdepth;

  int i0, i1, i2;
  mjtNum f0, f1, f2;

  int i, j, q, code, q1, q2, clcorner, n, m, k;
  int cle1, cle2, in, ax1, ax2, pax1, pax2, clface, nl, nf;

  n = 0;
  code = -1;
  margin2 = margin * margin;

  mji_sub3(tmp1, pos2, pos1);
  mji_mulMatTVec3(pos21, mat1, tmp1);

  mji_sub3(tmp1, pos1, pos2);
  mji_mulMatTVec3(pos12, mat2, tmp1);

  mju_mulMatTMat3(rot, mat1, mat2);
  mju_transpose(rott, rot, 3, 3);

  for (i = 0; i < 9; i++)
    rotabs[i] = mju_abs(rot[i]);
  for (i = 0; i < 9; i++)
    rottabs[i] = mju_abs(rott[i]);

  mji_mulMatVec3(plen2, rotabs, size2);
  mji_mulMatTVec3(plen1, rotabs, size1);

  for (i = 0, penetration = margin; i < 3; i++)
    penetration += size1[i] * 3 + size2[i] * 3;

  for (i = 0; i < 3; i++) {
    c1 = -mju_abs(pos21[i]) + size1[i] + plen2[i];
    c2 = -mju_abs(pos12[i]) + size2[i] + plen1[i];

    if (c1 < -margin || c2 < -margin)
      return 0;

    if (c1 < penetration) {
      penetration = c1;
      code = i + 3 * (pos21[i] < 0) + 0;
    }
    if (c2 < penetration) {
      penetration = c2;
      code = i + 3 * (pos12[i] < 0) + 6;
    }

    // printf("%24.16e %24.16e %d         %24.16e %d \n",c1,c2,i,penetration,code);
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      mju_zero3(tmp2);
      if (i == 0) {
        tmp2[1] = -rott[3 * j + 2];
        tmp2[2] = +rott[3 * j + 1];
      } else if (i == 1) {
        tmp2[0] = +rott[3 * j + 2];
        tmp2[2] = -rott[3 * j + 0];
      } else if (i == 2) {
        tmp2[0] = -rott[3 * j + 1];
        tmp2[1] = +rott[3 * j + 0];
      }

      c1 = mju_normalize3(tmp2);


      if (c1 < mjMINVAL)
        continue;

      c2 = mju_dot3(pos21, tmp2);

      c3 = 0;

      for (k = 0; k < 3; k++)
        if (k != i)
          c3 += size1[k] * mju_abs(tmp2[k]);
      for (k = 0; k < 3; k++)
        if (k != j)
          c3 += size2[k] * rotabs[3 * i + 3 - k - j] / c1;

      c3 -= mju_abs(c2);

      if (c3 < -margin)
        return 0;



      if (c3 < penetration * (1 - 1e-12))
      {
        penetration = c3;
        for (k = cle1 = 0; k < 3; k++)
          if (k != i)
            if ((tmp2[k] > 0) ^ (c2 < 0))
              cle1 += 1 << k;
        for (k = cle2 = 0; k < 3; k++)
          if (k != j)
            if ((rot[3 * i + 3 - k - j] > 0) ^ (c2 < 0) ^ ((k - j + 3) % 3 == 1))
              cle2 += 1 << k;

        code = 12 + i * 3 + j;
        mji_copy3(clnorm, tmp2);
        in = c2 < 0;
      }

      // printf("%24.16e %d      %24.16e %d\n",c3,12+i*3+j,penetration,code);
    }
  }


  // return 0;


  // printf("%d\n",code);

  if (code == -1)
    return 0;  // shouldn't happen

  if (code >= 12)
    goto edgeedge;


  q1 = code % 6;
  q2 = code / 6;

  // printf("%d %d\n",q1,q2);

  mju_zero(rotmore, 9);
  if (q1 == 0)
    rotmore[2] = -1, rotmore[4] = +1, rotmore[6] = +1;
  else if (q1 == 1)
    rotmore[0] = +1, rotmore[5] = -1, rotmore[7] = +1;
  else if (q1 == 2)
    rotmore[0] = +1, rotmore[4] = +1, rotmore[8] = +1;
  else if (q1 == 3)
    rotmore[2] = +1, rotmore[4] = +1, rotmore[6] = -1;
  else if (q1 == 4)
    rotmore[0] = +1, rotmore[5] = +1, rotmore[7] = -1;
  else if (q1 == 5)
    rotmore[0] = -1, rotmore[4] = +1, rotmore[8] = -1;

  i0 = 0;
  i1 = 1;
  i2 = 2;
  f0 = f1 = f2 = 1;

  if (q1 == 0) {
    i0 = 2;
    f0 = -1;
    i2 = 0;
  } else if (q1 == 1) {
    i1 = 2;
    f1 = -1;
    i2 = 1;
  } else if (q1 == 2) {
  } else if (q1 == 3) {
    i0 = 2;
    i2 = 0;
    f2 = -1;
  } else if (q1 == 4) {
    i1 = 2;
    i2 = 1;
    f2 = -1;
  } else if (q1 == 5) {
    f0 = -1;
    f2 = -1;
  }


#define rotaxis(vecres, vecin) \
{                              \
  vecres[0]=vecin[i0]*f0;      \
  vecres[1]=vecin[i1]*f1;      \
  vecres[2]=vecin[i2]*f2;      \
}
#define rotmatx(matres, matin)        \
{                                     \
  mji_scl3(matres+0, matin+i0*3, f0); \
  mji_scl3(matres+3, matin+i1*3, f1); \
  mji_scl3(matres+6, matin+i2*3, f2); \
}

  if (q2) {
    mju_mulMatMatT3(r, rotmore, rot);

    // mju_mulMatVec3(p,rotmore,pos12);
    // mju_mulMatVec3(tmp1,rotmore,size2);

    rotaxis(p, pos12);
    rotaxis(tmp1, size2);

    mji_copy3(s, size1);
  } else {
    // mju_mulMatMat(r,rotmore,rot,3,3,3);

    rotmatx(r, rot);

    // mju_mulMatVec3(p,rotmore,pos21);
    // mju_mulMatVec3(tmp1,rotmore,size1);

    rotaxis(p, pos21);
    rotaxis(tmp1, size1);

    mji_copy3(s, size2);
  }

  mju_transpose(rt, r, 3, 3);

  for (i = 0; i < 3; i++)
    ss[i] = mju_abs(tmp1[i]);

  lx = ss[0];
  ly = ss[1];
  hz = ss[2];
  p[2] -= hz;

  mji_copy3(lp, p);

  for (clcorner = 0, i = 0; i < 3; i++)
    if (r[6 + i] < 0)
      clcorner += 1 << i;

  mji_addToScl3(lp, rt + 0, s[0] * ((clcorner & 1) ? 1 : -1));
  mji_addToScl3(lp, rt + 3, s[1] * ((clcorner & 2) ? 1 : -1));
  mji_addToScl3(lp, rt + 6, s[2] * ((clcorner & 4) ? 1 : -1));

  m = k = 0;
  mji_copy3(pts[m++], lp);

  for (i = 0; i < 3; i++)
    if (mju_abs(r[6 + i]) < 0.5)
      mju_scl3(pts[m++], rt + 3 * i, s[i] * ((clcorner & (1 << i)) ? -2 : 2));

  mji_add3(pts[3], pts[0], pts[1]);
  mji_add3(pts[4], pts[0], pts[2]);
  mji_add3(pts[5], pts[3], pts[2]);

  if (m > 1)
  {
    mji_copy3(lines[k] + 0, pts[0]);
    mji_copy3(lines[k++] + 3, pts[1]);
  }
  if (m > 2)
  {
    mji_copy3(lines[k] + 0, pts[0]);
    mji_copy3(lines[k++] + 3, pts[2]);
    mji_copy3(lines[k] + 0, pts[3]);
    mji_copy3(lines[k++] + 3, pts[2]);
    mji_copy3(lines[k] + 0, pts[4]);
    mji_copy3(lines[k++] + 3, pts[1]);
  }

  for (i = 0; i < k; i++) {
    for (q = 0; q < 2; q++) {
      a = lines[i][0 + q];
      b = lines[i][3 + q];
      c = lines[i][1 - q];
      d = lines[i][4 - q];

      if (mju_abs(b) > mjMINVAL) {
        for (j = -1; j <= 1; j += 2) {
          l = ss[q] * j;
          c1 = (l - a) * (1 / b);
          if (c1 < 0 || c1 > 1)
            continue;
          c2 = c + d * c1;
          if (mju_abs(c2) > ss[1 - q])
            continue;

          if (n < mjMAXCONPAIR) {
            mji_copy3(points[n], lines[i]);
            mji_addToScl3(points[n++], lines[i] + 3, c1);
          }
        }
      }
    }
  }


  a = pts[1][0];
  b = pts[2][0];
  c = pts[1][1];
  d = pts[2][1];
  c1 = a * d - b * c;


  if (m > 2) {
    for (i = 0; i < 4; i++) {
      llx = i / 2 ? lx : -lx;
      lly = i % 2 ? ly : -ly;

      x = llx - pts[0][0];
      y = lly - pts[0][1];

      u = (x * d - y * b) * (1 / c1);
      v = (y * a - x * c) * (1 / c1);
      if (u <= 0 || v <= 0 || u >= 1 || v >= 1)
        continue;

      if (n < mjMAXCONPAIR) {
        points[n][0] = llx;
        points[n][1] = lly;
        points[n][2] = (pts[0][2] + u * pts[1][2] + v * pts[2][2]);
        n++;
      }
    }
  }

  for (i = 0; i < (1 << (m - 1)); i++) {
    mji_copy3(tmp1, pts[i == 0 ? 0 : i + 2]);


    if (i)
      if (tmp1[0] <= -lx || tmp1[0] >= lx)
        continue;
    if (i)
      if (tmp1[1] <= -ly || tmp1[1] >= ly)
        continue;

    if (n < mjMAXCONPAIR) {
      mji_copy3(points[n++], tmp1);
    }
  }


  m = n;
  n = 0;

  for (i = 0; i < m; i++) {
    if (points[i][2] > margin)
      continue;
    if (n != i) mji_copy3(points[n], points[i]);

    depth[n] = points[n][2];
    points[n][2] *= 0.5;

    n++;
  }


  mju_mulMatMatT3(r, q2 ? mat2 : mat1, rotmore);
  mju_copy3(p, q2 ? pos2 : pos1);

  tmp2[0] = (q2 ? -1 : 1) * r[2];
  tmp2[1] = (q2 ? -1 : 1) * r[5];
  tmp2[2] = (q2 ? -1 : 1) * r[8];

  mji_copy3(con[0].normal, tmp2);
  mji_zero3(con[0].tangent);




  for (i = 0; i < n; i++) {
    con[i].dist = 2 * points[i][2];
    points[i][2] += hz;

    mji_mulMatVec3(tmp2, r, points[i]);
    mji_add3(con[i].pos, tmp2, p);

    if (i) {
      mji_copy3(con[i].normal, con[0].normal);
      mji_zero3(con[i].tangent);
    }
  }


  // printf("Path1:  %d\n",n);


  return n;

edgeedge:


  code -= 12;

  q1 = code / 3;
  q2 = code % 3;



  if (q2 == 0)
    ax1 = 1, ax2 = 2;
  if (q2 == 1)
    ax1 = 0, ax2 = 2;
  if (q2 == 2)
    ax1 = 1, ax2 = 0;
  if (q1 == 0)
    pax1 = 1, pax2 = 2;
  if (q1 == 1)
    pax1 = 0, pax2 = 2;
  if (q1 == 2)
    pax1 = 1, pax2 = 0;

  // printf("%lf %lf   %lf %lf\n",rot[ 3*q1+ ax1],rot [3*q1+ ax2],rott[3*q2+pax1],rott[3*q2+pax2]);
  // printf("%lf %lf\n",mju_dot3(clnorm,rott+3*ax1),mju_dot3(clnorm,rott+3*ax2));

  if (rotabs [3 * q1 + ax1] < rotabs [3 * q1 + ax2]) {
    ax1 = ax2;
    ax2 = 3 - q2 - ax1;
  }
  if (rottabs[3 * q2 + pax1] < rottabs[3 * q2 + pax2]) {
    pax1 = pax2;
    pax2 = 3 - q1 - pax1;
  }

  if (cle1 & (1 << pax2))
    clface = pax2;
  else
    clface = pax2 + 3;


  // printf("%lf - %d %d %d %d   %d %d     %d %d %d %d %d\n",
  //        penetration,cle1,cle2,code,in,q1,q2,clface,ax1,ax2,pax1,pax2);


  mju_zero(rotmore, 9);
  if (clface == 0)
    rotmore[2] = -1, rotmore[4] = +1, rotmore[6] = +1;
  else if (clface == 1)
    rotmore[0] = +1, rotmore[5] = -1, rotmore[7] = +1;
  else if (clface == 2)
    rotmore[0] = +1, rotmore[4] = +1, rotmore[8] = +1;
  else if (clface == 3)
    rotmore[2] = +1, rotmore[4] = +1, rotmore[6] = -1;
  else if (clface == 4)
    rotmore[0] = +1, rotmore[5] = +1, rotmore[7] = -1;
  else if (clface == 5)
    rotmore[0] = -1, rotmore[4] = +1, rotmore[8] = -1;


  i0 = 0;
  i1 = 1;
  i2 = 2;
  f0 = f1 = f2 = 1;

  if (clface == 0) {
    i0 = 2;
    f0 = -1;
    i2 = 0;
  } else if (clface == 1) {
    i1 = 2;
    f1 = -1;
    i2 = 1;
  } else if (clface == 2) {
  } else if (clface == 3) {
    i0 = 2;
    i2 = 0;
    f2 = -1;
  } else if (clface == 4) {
    i1 = 2;
    i2 = 1;
    f2 = -1;
  } else if (clface == 5) {
    f0 = -1;
    f2 = -1;
  }

  // mju_mulMatVec3(p,rotmore,pos21);
  // mju_mulMatVec3(rnorm,rotmore,clnorm);
  rotaxis(p, pos21);
  rotaxis(rnorm, clnorm);

  // print("rnorm",rnorm);

  // mju_mulMatMat(r,rotmore,rot,3,3,3);
  rotmatx(r, rot);

  mji_mulMatTVec3(tmp1, rotmore, size1);
  for (i = 0; i < 3; i++)
    s[i] = mju_abs(tmp1[i]);

  mju_transpose(rt, r, 3, 3);


  lx = s[0];
  ly = s[1];
  hz = s[2];
  p[2] -= hz;


  n = 0;
  mji_copy3(points[n], p);
  mji_addToScl3(points[n], rt + 3 * ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? 1 : -1));
  mji_addToScl3(points[n], rt + 3 * ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1 : -1));
  mji_copy3(points[n + 1], points[n]);
  mji_addToScl3(points[n], rt + 3 * q2, size2[q2]);
  n = 1;
  mji_addToScl3(points[n], rt + 3 * q2, -size2[q2]);
  n = 2;


  mji_copy3(points[n], p);
  mji_addToScl3(points[n], rt + 3 * ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? -1 : 1));
  mji_addToScl3(points[n], rt + 3 * ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1 : -1));
  mji_copy3(points[n + 1], points[n]);
  mji_addToScl3(points[n], rt + 3 * q2, size2[q2]);
  n = 3;
  mji_addToScl3(points[n], rt + 3 * q2, -size2[q2]);
  n = 4;


  mji_copy3(axi[0], points[0]);
  mji_sub3(axi[1], points[1], points[0]);
  mji_sub3(axi[2], points[2], points[0]);


  if (mju_abs(rnorm[2]) < mjMINVAL)
    return 0;  // shouldn't happen

  innorm = (1 / rnorm[2]) * (in ? -1 : 1);
  // printf("%lf\n",innorm);

  for (i = 0; i < 4; i++)
  {
    c1 = -points[i][2] * (1 / rnorm[2]);

    mji_copy3(pu[i], points[i]);

    mji_addToScl3(points[i], rnorm, c1);

    // ppts[i][0]=points[i][0];
    // ppts[i][1]=points[i][1];
    ppts2[i][0] = points[i][0];
    ppts2[i][1] = points[i][1];
  }


  mji_copy3(pts[0], points[0]);
  mji_sub3(pts[1], points[1], points[0]);
  mji_sub3(pts[2], points[2], points[0]);

  m = 3;
  k = 0;
  n = 0;


  if (m > 1) {
    mji_copy3(lines[k] + 0, pts[0]);
    mji_copy3(lines[k] + 3, pts[1]);
    mji_copy3(linesu[k] + 0, axi[0]);
    mji_copy3(linesu[k++] + 3, axi[1]);
  }
  if (m > 2) {
    mji_copy3(lines[k] + 0, pts[0]);
    mji_copy3(lines[k] + 3, pts[2]);
    mji_copy3(linesu[k] + 0, axi[0]);
    mji_copy3(linesu[k++] + 3, axi[2]);

    mji_add3(lines[k] + 0, pts[0], pts[1]);
    mji_copy3(lines[k] + 3, pts[2]);
    mji_add3(linesu[k] + 0, axi[0], axi[1]);
    mji_copy3(linesu[k++] + 3, axi[2]);

    mji_add3(lines[k] + 0, pts[0], pts[2]);
    mji_copy3(lines[k] + 3, pts[1]);
    mji_add3(linesu[k] + 0, axi[0], axi[2]);
    mji_copy3(linesu[k++] + 3, axi[1]);
  }

  for (i = 0; i < k; i++) {
    for (q = 0; q < 2; q++) {
      a = lines[i][0 + q];
      b = lines[i][3 + q];
      c = lines[i][1 - q];
      d = lines[i][4 - q];

      if (mju_abs(b) > mjMINVAL) {
        for (j = -1; j <= 1; j += 2) {
          if (n < mjMAXCONPAIR) {
            l = s[q] * j;
            c1 = (l - a) * (1 / b);
            if (c1 < 0 || c1 > 1)
              continue;
            c2 = c + d * c1;
            if (mju_abs(c2) > s[1 - q])
              continue;

            if ((linesu[i][2] + linesu[i][5]*c1)*innorm > margin)
              continue;

            mji_scl3(points[n], linesu[i], 0.5);
            mji_addToScl3(points[n], linesu[i] + 3, 0.5 * c1);
            points[n][0 + q] += 0.5 * l;
            points[n][1 - q] += 0.5 * c2;
            depth[n] = points[n][2] * innorm * 2;
            n++;
          }
        }
      }
    }
  }

  nl = n;

  a = pts[1][0];
  b = pts[2][0];
  c = pts[1][1];
  d = pts[2][1];
  c1 = a * d - b * c;

  for (i = 0; i < 4; i++) {
    if (n < mjMAXCONPAIR) {
      llx = i / 2 ? lx : -lx;
      lly = i % 2 ? ly : -ly;

      x = llx - pts[0][0];
      y = lly - pts[0][1];

      u = (x * d - y * b) * (1 / c1);
      v = (y * a - x * c) * (1 / c1);

      if (nl == 0) {
        if ((u < 0 || u > 1) && (v < 0 || v > 1))
          continue;
      } else {
        if ((u < 0 || u > 1 || v < 0 || v > 1))
          continue;
      }

      if (u < 0)
        u = 0;
      if (u > 1)
        u = 1;
      if (v < 0)
        v = 0;
      if (v > 1)
        v = 1;


      mji_scl3(tmp1, pu[0], 1 - u - v);
      mji_addToScl3(tmp1, pu[1], u);
      mji_addToScl3(tmp1, pu[2], v);

      points[n][0] = llx;
      points[n][1] = lly;
      points[n][2] = 0;

      mji_sub3(tmp2, points[n], tmp1);

      c1 = mju_dot3(tmp2, tmp2);
      if (tmp1[2] > 0)
        if (c1 > margin2)
          continue;

      mji_addTo3(points[n], tmp1);
      mju_scl3(points[n], points[n], 0.5);

      depth[n] = sqrt(c1) * (tmp1[2] < 0 ? -1 : 1);
      n++;
    }
  }

  nf = n;

  for (i = 0; i < 4; i++) {
    if (n < mjMAXCONPAIR) {
      x = ppts2[i][0];
      y = ppts2[i][1];

      if (nl == 0) {
        if (nf == 0) {
        } else {
          if (x < -lx || x > lx)
            if (y < -ly || y > ly)
              continue;
        }
      } else {
        if (x < -lx || x > lx || y < -ly || y > ly)
          continue;
      }

      for (c1 = 0, j = 0; j < 2; j++)
        if (ppts2[i][j] < -s[j])
          c1 += (ppts2[i][j] + s[j]) * (ppts2[i][j] + s[j]);
        else if (ppts2[i][j] > s[j])
          c1 += (ppts2[i][j] - s[j]) * (ppts2[i][j] - s[j]);

      c1 += pu[i][2] * innorm * pu[i][2] * innorm;

      if (pu[i][2] > 0)
        if (c1 > margin2)
          continue;


      tmp1[0] = ppts2[i][0] * 0.5;
      tmp1[1] = ppts2[i][1] * 0.5;
      tmp1[2] = 0;

      for (j = 0; j < 2; j++) {
        if (ppts2[i][j] < -s[j])
          tmp1[j] = -s[j] * 0.5;
        else if (ppts2[i][j] > s[j])
          tmp1[j] = +s[j] * 0.5;
      }
      mji_addToScl3(tmp1, pu[i], 0.5);
      mji_copy3(points[n], tmp1);

      depth[n] = sqrt(c1) * (pu[i][2] < 0 ? -1 : 1);
      n++;
    }
  }

  mju_mulMatMatT3(r, mat1, rotmore);

  mji_mulMatVec3(tmp1, r, rnorm);

  mji_scl3(con[0].normal, tmp1, in ? -1 : 1);
  mji_zero3(con[0].tangent);


  // no contact can be deeper than the support overlap along the separating axis: clipping
  // against a grazing face can synthesize spurious points with arbitrarily large depth;
  // the slack covers rounding error so the deepest legitimate point is never rejected
  maxdepth = mju_max(0, penetration);
  maxdepth += margin + mjDEPTHSLACKREL * maxdepth +
              mjDEPTHSLACKABS * (size1[0] + size1[1] + size1[2] +
                                 size2[0] + size2[1] + size2[2]);

  for (i = 0, m = 0; i < n; i++) {
    if (depth[i] < -maxdepth)
      continue;

    con[m].dist = depth[i];
    points[i][2] += hz;

    mji_mulMatVec3(tmp2, r, points[i]);

    mji_add3(con[m].pos, tmp2, pos1);

    mji_copy3(con[m].normal, con[0].normal);
    mji_zero3(con[m].tangent);
    m++;
  }

  return m;

#undef rotaxis
#undef rotmatx
}

// legacy box : box
static int mjc_BoxBoxLegacy(const mjModel* m, mjData* d, mjPreContact* con,
                            int g1, int g2, mjtNum margin) {
  mjPreContact tmp[mjMAXCONPAIR];
  int num = boxboxLegacyRaw(m, d, tmp, g1, g2, margin);

  // -1: bad, 0: good
  int dupe[mjMAXCONPAIR] = {0};


  // get box info
  const mjtNum* pos1 =  d->geom_xpos + 3 * g1;
  const mjtNum* mat1 =  d->geom_xmat + 9 * g1;
  const mjtNum* size1 = m->geom_size + 3 * g1;
  const mjtNum* pos2 =  d->geom_xpos + 3 * g2;
  const mjtNum* mat2 =  d->geom_xmat + 9 * g2;
  const mjtNum* size2 = m->geom_size + 3 * g2;

  // find bad: contacts outside one of the boxes
  for (int i=0; i < num; i++) {
    // box sizes with margin
    mjtNum sz1[3] = {size1[0] + margin, size1[1] + margin, size1[2] + margin};
    mjtNum sz2[3] = {size2[0] + margin, size2[1] + margin, size2[2] + margin};

    // relative distance from surface (1%) outside of which box-box contacts are removed
    static mjtNum kRemoveRatio = 1.01;

    // is the contact outside: 1, inside: -1, within the removal width: 0
    int out1 = mju_outsideBox(tmp[i].pos, pos1, mat1, sz1, kRemoveRatio);
    int out2 = mju_outsideBox(tmp[i].pos, pos2, mat2, sz2, kRemoveRatio);

    // mark as bad if outside one box and not inside the other box
    if ((out1 == 1 && out2 != -1) || (out2 == 1 && out1 != -1)) {
      dupe[i] = -1;
    }
  }

  // find duplicates
  for (int i=0; i < num-1; i++) {
    if (dupe[i] == -1) {
      continue;  // already marked bad: skip
    }
    for (int j=i+1; j < num; j++) {
      if (dupe[j] == -1) {
        continue;  // already marked bad: skip
      }
      if (tmp[i].pos[0] == tmp[j].pos[0] &&
          tmp[i].pos[1] == tmp[j].pos[1] &&
          tmp[i].pos[2] == tmp[j].pos[2]) {
        dupe[i] = -1;
        break;
      }
    }
  }

  // consolidate good
  int ncon = 0;
  for (int j=0; j < num; j++) {
    if (dupe[j] == 0) {
      con[ncon++] = tmp[j];
      if (ncon >= 8) {
        break;
      }
    }
  }

  return ncon;
}

#undef mjDEPTHSLACKREL
#undef mjDEPTHSLACKABS

// box : box
int mjc_BoxBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  // collider selection via mjOption.boxbox: the legacy implementation and the general
  // convex pipeline (GJK/EPA) are available for comparison
  if (m->opt.boxbox == mjBOXBOX_CONVEX) {
    return mjc_Convex(m, d, con, g1, g2, margin);
  }
  if (m->opt.boxbox == mjBOXBOX_LEGACY) {
    return mjc_BoxBoxLegacy(m, d, con, g1, g2, margin);
  }

  const mjtNum* pos1 = d->geom_xpos + 3*g1;
  const mjtNum* pos2 = d->geom_xpos + 3*g2;
  const mjtNum* mat1 = d->geom_xmat + 9*g1;
  const mjtNum* mat2 = d->geom_xmat + 9*g2;
  const mjtNum* size1 = m->geom_size + 3*g1;
  const mjtNum* size2 = m->geom_size + 3*g2;

  // rot: box2 axes in box1 frame (columns); pos21: box2 center in box1 frame;
  // pos12: box1 center in box2 frame
  mjtNum rot[9], rotabs[9], pos21[3], pos12[3], tmp[3];
  mji_sub3(tmp, pos2, pos1);
  mji_mulMatTVec3(pos21, mat1, tmp);
  mji_sub3(tmp, pos1, pos2);
  mji_mulMatTVec3(pos12, mat2, tmp);
  mju_mulMatTMat3(rot, mat1, mat2);
  for (int i = 0; i < 9; i++) {
    rotabs[i] = mju_abs(rot[i]);
  }

  //------------------------------ stage 1: separating-axis test

  // best separation so far (most positive; negative = penetration), and the winning axis:
  // code 0..2 face of box1, 3..5 face of box2, >= 6 edge pair (i, j) as 6 + 3*i + j
  mjtNum sep_best = -mjMAXVAL;
  mjtNum sep_face = -mjMAXVAL;
  int code = -1;

  // face axes of box1: candidate normal is axis i of box1
  for (int i = 0; i < 3; i++) {
    mjtNum radius2 = rotabs[3*i+0]*size2[0] + rotabs[3*i+1]*size2[1] + rotabs[3*i+2]*size2[2];
    mjtNum sep = mju_abs(pos21[i]) - size1[i] - radius2;
    if (sep > margin) {
      return 0;
    }
    if (sep > sep_best) {
      sep_best = sep;
      code = i;
    }
  }

  // face axes of box2: candidate normal is axis j of box2
  for (int j = 0; j < 3; j++) {
    mjtNum radius1 = rotabs[0+j]*size1[0] + rotabs[3+j]*size1[1] + rotabs[6+j]*size1[2];
    mjtNum sep = mju_abs(pos12[j]) - size2[j] - radius1;
    if (sep > margin) {
      return 0;
    }
    if (sep > sep_best) {
      sep_best = sep;
      code = 3 + j;
    }
  }
  sep_face = sep_best;
  int code_face = code;

  // edge-cross axes: candidate direction is axis i of box1 crossed with axis j of box2
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // cross product of e_i with column j of rot, in box1 frame; component i is zero
      int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
      mjtNum ax1 = -rot[3*i2+j];
      mjtNum ax2 = rot[3*i1+j];

      // the cross product of two unit vectors has norm sin(angle); for nearly parallel
      // edges the components above are pure cancellation noise and the direction is
      // meaningless, so require sin(angle) well above rounding; the skipped axes are
      // covered by the face normals, which the cross product converges to as the angle
      // vanishes
      mjtNum norm2 = ax1*ax1 + ax2*ax2;
      if (norm2 < 1e-16) {
        continue;
      }
      mjtNum inv = 1/mju_sqrt(norm2);
      ax1 *= inv;
      ax2 *= inv;

      // support radius of box1: component i of axis is zero by construction
      mjtNum radius1 = size1[i1]*mju_abs(ax1) + size1[i2]*mju_abs(ax2);

      // support radius of box2: transform axis to box2 frame; component j is zero there,
      // and only components i1, i2 of the axis are nonzero here
      int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      mjtNum a2_1 = ax1*rot[3*i1+j1] + ax2*rot[3*i2+j1];
      mjtNum a2_2 = ax1*rot[3*i1+j2] + ax2*rot[3*i2+j2];
      mjtNum radius2 = size2[j1]*mju_abs(a2_1) + size2[j2]*mju_abs(a2_2);

      mjtNum sep = mju_abs(ax1*pos21[i1] + ax2*pos21[i2]) - radius1 - radius2;
      if (sep > margin) {
        return 0;
      }

      // an edge axis must beat the best face axis by a bias-scaled amount: on exact ties
      // the face manifold (multiple points) is strictly better for the solver
      if (sep - mjBOXBOX_EDGEBIAS*mju_abs(sep) > sep_best && sep > sep_face) {
        sep_best = sep;
        code = 6 + 3*i + j;
      }
    }
  }

  if (code < 0) {
    return 0;  // cannot happen: some face axis always sets code
  }

  // a winning edge axis nearly parallel to the best face axis (within ~8 degrees)
  // duplicates it: the face manifold covers the same contact with multiple points, and
  // resting stacks flip between the two codes by rounding noise if the near-tie is
  // allowed to alternate. The face is substituted unless the edge is better by five
  // percent of the face depth (ODE's classic fudge): resting-stack energy degrades
  // continuously as this margin shrinks, while the depth cost of the substitution is
  // bounded by the same five percent. Substituting after the search, rather than
  // filtering during it, prevents a worse non-aliasing edge from stealing the contact
  // that the substitution meant to give to the face.
  if (code >= 6) {
    int i = (code - 6) / 3;
    int j = (code - 6) % 3;
    int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    mjtNum axis[3];
    axis[i] = 0;
    axis[i1] = -rot[3*i2+j];
    axis[i2] = rot[3*i1+j];
    mju_normalize3(axis);
    mjtNum face_dot;
    if (code_face < 3) {
      face_dot = mju_abs(axis[code_face]);
    } else {
      int f = code_face - 3;
      face_dot = mju_abs(axis[0]*rot[0+f] + axis[1]*rot[3+f] + axis[2]*rot[6+f]);
    }
    if (face_dot > 0.99 && sep_best < sep_face + 0.05*mju_abs(sep_face) + mjMINVAL) {
      code = code_face;
      sep_best = sep_face;
    }
  }

  //------------------------------ stage 2a: edge-edge contact

  if (code >= 6) {
    int i = (code - 6) / 3;
    int j = (code - 6) % 3;
    int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
    int j1 = (j + 1) % 3, j2 = (j + 2) % 3;

    // unit separating axis in box1 frame, oriented from box1 toward box2
    mjtNum axis[3];
    axis[i] = 0;
    axis[i1] = -rot[3*i2+j];
    axis[i2] = rot[3*i1+j];
    mju_normalize3(axis);
    if (mju_dot3(axis, pos21) < 0) {
      mji_scl3(axis, axis, -1);
    }

    // supporting edges: the box1 edge runs along e_i at a corner selected by the axis
    // signs in (i1, i2); the box2 edge runs along column j at a corner selected by the
    // signs of the axis in box2 coordinates. A near-zero component makes the sign choice
    // meaningless -- both edges support the axis -- and rounding can pick the wrong one,
    // producing witness points on the wrong side of the box. Enumerate both signs for any
    // ambiguous component (at most one per box) and keep the closest witness pair.
    mjtNum a2[3] = {
      axis[0]*rot[0+0] + axis[1]*rot[3+0] + axis[2]*rot[6+0],
      axis[0]*rot[0+1] + axis[1]*rot[3+1] + axis[2]*rot[6+1],
      axis[0]*rot[0+2] + axis[1]*rot[3+2] + axis[2]*rot[6+2],
    };
    const mjtNum ambig = 1e-9;
    int amb1 = -1, amb2 = -1;
    if (mju_abs(axis[i1]) < ambig) amb1 = i1;
    else if (mju_abs(axis[i2]) < ambig) amb1 = i2;
    if (mju_abs(a2[j1]) < ambig) amb2 = j1;
    else if (mju_abs(a2[j2]) < ambig) amb2 = j2;

    mjtNum d2[3] = {rot[0+j], rot[3+j], rot[6+j]};
    mjtNum b = d2[i];  // d1 . d2, with d1 = e_i
    mjtNum denom = 1 - b*b;

    mjtNum w1[3], w2[3];
    mjtNum best_d2 = mjMAXVAL;
    for (int v1 = 0; v1 < (amb1 >= 0 ? 2 : 1); v1++) {
      for (int v2 = 0; v2 < (amb2 >= 0 ? 2 : 1); v2++) {
        // corner of the box1 edge: support along +axis, ambiguous component flipped by v1
        mjtNum c1[3];
        c1[i] = 0;
        c1[i1] = axis[i1] >= 0 ? size1[i1] : -size1[i1];
        c1[i2] = axis[i2] >= 0 ? size1[i2] : -size1[i2];
        if (amb1 >= 0 && v1) c1[amb1] = -c1[amb1];

        // corner of the box2 edge: support along -axis in box2 coordinates
        mjtNum cc[3];
        cc[j] = 0;
        cc[j1] = a2[j1] >= 0 ? -size2[j1] : size2[j1];
        cc[j2] = a2[j2] >= 0 ? -size2[j2] : size2[j2];
        if (amb2 >= 0 && v2) cc[amb2] = -cc[amb2];
        mjtNum c2[3];
        mji_mulMatVec3(c2, rot, cc);
        mji_addTo3(c2, pos21);

        // closest points between the two edge segments (directions are unit vectors)
        mjtNum e[3];
        mji_sub3(e, c2, c1);
        mjtNum d1e = e[i];  // d1 . e
        mjtNum d2e = mju_dot3(d2, e);
        mjtNum s = denom < mjMINVAL ? 0 : (d1e - b*d2e) / denom;

        // clamp into the segments, letting each clamp re-solve the other parameter
        s = mju_clip(s, -size1[i], size1[i]);
        mjtNum t = mju_clip(b*s - d2e, -size2[j], size2[j]);
        s = mju_clip(d1e + b*t, -size1[i], size1[i]);

        mjtNum p1[3], p2[3], gap[3];
        mji_copy3(p1, c1);
        p1[i] += s;
        mji_copy3(p2, c2);
        mji_addToScl3(p2, d2, t);
        mji_sub3(gap, p2, p1);
        mjtNum g2 = mju_dot3(gap, gap);
        if (g2 < best_d2) {
          best_d2 = g2;
          mji_copy3(w1, p1);
          mji_copy3(w2, p2);
        }
      }
    }

    // signed distance along the axis
    mjtNum gap[3];
    mji_sub3(gap, w2, w1);
    mjtNum dist = mju_dot3(gap, axis);
    if (dist > margin) {
      return 0;
    }

    // contact at the midpoint of the witness pair: for penetrating edges this is inside both
    // boxes; in the margin band it is midway between the two surfaces
    mjtNum mid[3];
    mji_add3(mid, w1, w2);
    mji_scl3(mid, mid, 0.5);

    con[0].dist = dist;
    mji_mulMatVec3(tmp, mat1, mid);
    mji_add3(con[0].pos, tmp, pos1);
    mji_mulMatVec3(con[0].normal, mat1, axis);
    mji_zero3(con[0].tangent);
    return 1;
  }

  //------------------------------ stage 2b: face contact

  // reference box: owner of the winning face; incident box: the other one
  int ref1 = code < 3;              // is box1 the reference?
  int a = ref1 ? code : code - 3;   // face axis of the reference box
  const mjtNum* sizeref = ref1 ? size1 : size2;
  const mjtNum* sizeinc = ref1 ? size2 : size1;
  const mjtNum* posref = ref1 ? pos1 : pos2;
  const mjtNum* matref = ref1 ? mat1 : mat2;
  const mjtNum* posoi = ref1 ? pos21 : pos12;  // incident center in reference frame

  // incident box axes in reference frame: rot maps box2 to box1, transpose maps box1 to box2;
  // rinc(r, c) = component r of incident axis c, in reference frame
  mjtNum rinc[9];
  if (ref1) {
    mju_copy(rinc, rot, 9);
  } else {
    mju_transpose(rinc, rot, 3, 3);
  }

  // face direction: +1 if the incident box lies along +a, else -1
  mjtNum sgn = posoi[a] >= 0 ? 1 : -1;

  // incident face: the face of the incident box most opposed to the reference face normal
  int binc = 0;
  for (int k = 1; k < 3; k++) {
    if (mju_abs(rinc[3*a+k]) > mju_abs(rinc[3*a+binc])) {
      binc = k;
    }
  }
  mjtNum tinc = sgn*rinc[3*a+binc] > 0 ? -1 : 1;  // sign making the incident normal oppose

  // corners of the incident face in reference frame, cyclic winding; the in-plane
  // coordinates are (x, y) = the two non-a reference axes, z is the signed distance
  // above the reference face plane (negative = inside the reference box)
  int ax = (a + 1) % 3, ay = (a + 2) % 3;
  int bu = (binc + 1) % 3, bv = (binc + 2) % 3;
  mjtNum poly[2][mjBOXBOX_MAXVERT][3];

  // face center and in-face half-edge offsets, in the projected (x, y, z) coordinates
  mjtNum cx[3], du[3], dv[3];
  for (int r = 0; r < 3; r++) {
    int c = r == 0 ? ax : (r == 1 ? ay : a);
    cx[r] = posoi[c] + tinc*sizeinc[binc]*rinc[3*c+binc];
    du[r] = sizeinc[bu]*rinc[3*c+bu];
    dv[r] = sizeinc[bv]*rinc[3*c+bv];
  }
  cx[2] = sgn*cx[2] - sizeref[a];
  du[2] *= sgn;
  dv[2] *= sgn;
  static const mjtNum corner_sign[4][2] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
  for (int k = 0; k < 4; k++) {
    mjtNum su = corner_sign[k][0], sv = corner_sign[k][1];
    poly[0][k][0] = cx[0] + su*du[0] + sv*dv[0];
    poly[0][k][1] = cx[1] + su*du[1] + sv*dv[1];
    poly[0][k][2] = cx[2] + su*du[2] + sv*dv[2];
  }

  // clip against the four side planes of the reference face; the buffers swap only on
  // passes that actually clip
  int nvert = 4;
  mjtNum (*cur)[3] = poly[0];
  mjtNum (*spare)[3] = poly[1];
  nvert = clipHalfPlane(nvert, &cur, &spare, 0, 1, sizeref[ax]);
  nvert = clipHalfPlane(nvert, &cur, &spare, 0, -1, sizeref[ax]);
  nvert = clipHalfPlane(nvert, &cur, &spare, 1, 1, sizeref[ay]);
  nvert = clipHalfPlane(nvert, &cur, &spare, 1, -1, sizeref[ay]);
  const mjtNum (*clipped)[3] = cur;

  // accept vertices within the margin band, dropping near-duplicates produced by clipping
  // at polygon corners; duplicate radius is relative to the reference face scale
  mjtNum accepted[mjBOXBOX_MAXVERT][3];
  int naccept = 0;
  mjtNum dupe2 = 1e-14*(sizeref[ax]*sizeref[ax] + sizeref[ay]*sizeref[ay]);
  for (int k = 0; k < nvert; k++) {
    if (clipped[k][2] > margin) {
      continue;
    }
    int dupe = 0;
    for (int q = 0; q < naccept; q++) {
      mjtNum dx = accepted[q][0] - clipped[k][0];
      mjtNum dy = accepted[q][1] - clipped[k][1];
      if (dx*dx + dy*dy < dupe2) {
        dupe = 1;
        break;
      }
    }
    if (!dupe) {
      mji_copy3(accepted[naccept++], clipped[k]);
    }
  }
  if (naccept == 0) {
    return 0;
  }

  // select at most mjBOXBOX_MAXCON contacts
  int sel[mjBOXBOX_MAXCON];
  int ncon = reduceManifold(naccept, accepted, sel);

  // world normal points from geom1 to geom2: along +sgn*a of the reference frame when box1
  // is the reference, opposite when box2 is
  mjtNum normal[3];
  mjtNum nsign = ref1 ? sgn : -sgn;
  normal[0] = nsign*matref[3*0+a];
  normal[1] = nsign*matref[3*1+a];
  normal[2] = nsign*matref[3*2+a];

  for (int k = 0; k < ncon; k++) {
    const mjtNum* v = accepted[sel[k]];

    // contact position: on the clipped incident polygon in (x, y), midway between the
    // reference face plane and the incident surface along the face axis
    mjtNum posc[3];
    posc[ax] = v[0];
    posc[ay] = v[1];
    posc[a] = sgn*(sizeref[a] + 0.5*v[2]);

    con[k].dist = v[2];
    mji_mulMatVec3(tmp, matref, posc);
    mji_add3(con[k].pos, tmp, posref);
    mji_copy3(con[k].normal, normal);
    mji_zero3(con[k].tangent);
  }
  return ncon;
}
