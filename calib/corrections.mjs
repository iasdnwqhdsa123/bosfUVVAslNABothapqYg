import { predict } from "./gbm.js";
const RELIEF_DEG = 0.15;
const CU_DEPTH = 300;
const CLAMPS = { ws: [0.5, 2], top: [0.5, 2], zi: [0.5, 2], cb: 600, cu: [0.3, 0.7] };
const doyOf = (day) => {
  const [y, m, d] = day.split("-").map(Number);
  return Math.round((Date.UTC(y, m - 1, d) - Date.UTC(y, 0, 0)) / 864e5);
};
const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
function reliefGrid(elev, nx, ny, step) {
  const r = Math.round(RELIEF_DEG / step), out = new Float32Array(nx * ny).fill(NaN);
  for (let iy = 0; iy < ny; iy++)
    for (let ix = 0; ix < nx; ix++) {
      let lo = Infinity, hi = -Infinity;
      for (let y = Math.max(0, iy - r); y <= Math.min(ny - 1, iy + r); y++)
        for (let x = Math.max(0, ix - r); x <= Math.min(nx - 1, ix + r); x++) {
          const e = elev[y * nx + x];
          if (Number.isFinite(e)) lo = Math.min(lo, e), hi = Math.max(hi, e);
        }
      if (hi >= lo) out[iy * nx + ix] = hi - lo;
    }
  return out;
}
function applyCorrections(fields, meta, corr) {
  const { names, nx, ny, nh } = fields, NF = names.length, N = nx * ny;
  const A = fields.data ?? fields.all, out = new Float32Array(A);
  const fi = Object.fromEntries(names.map((n, i) => [n, i]));
  const at = (h, name, iy, ix) => fi[name] == null ? NaN : A[(h * NF + fi[name]) * N + iy * nx + ix];
  const put = (h, name, iy, ix, v) => out[(h * NF + fi[name]) * N + iy * nx + ix] = v;
  const g = fields.grid ?? meta.grid, doy = doyOf(meta.day), ds = Math.sin(2 * Math.PI * doy / 365.25), dc = Math.cos(2 * Math.PI * doy / 365.25);
  const mid = meta.modelIndex ?? corr.model_ids?.[meta.model] ?? NaN;
  const hourMs = (h) => {
    const st = fields.stamps?.[h];
    if (st == null) return typeof meta.hours[h] === "number" ? meta.hours[h] : meta.hours[h].utc;
    return meta.stampUtc?.[st] ?? Date.UTC(+st.slice(0, 4), +st.slice(4, 6) - 1, +st.slice(6, 8), +st.slice(8, 10));
  };
  const relief = fi.elevation != null ? reliefGrid(A.subarray(fi.elevation * N, fi.elevation * N + N), nx, ny, g.step) : null;
  const F = corr.fields, cl = { ...CLAMPS, ...corr.clamps || {} }, cuDepth = corr.cu_depth ?? CU_DEPTH;
  const feat = Object.fromEntries(corr.features.map((k, i) => [k, i]));
  const x = new Array(corr.features.length);
  const row = (m) => m.features.map((k) => x[feat[k]]);
  for (let h = 0; h < nh; h++) {
    const utcH = (hourMs(h) / 36e5 % 24 + 24) % 24;
    for (let iy = 0; iy < ny; iy++) {
      const lat = g.north - iy * g.step;
      for (let ix = 0; ix < nx; ix++) {
        const ws = at(h, "wstar", iy, ix);
        if (!Number.isFinite(ws)) continue;
        const lon = g.west + ix * g.step;
        for (const k of corr.features) {
          let v;
          if (k === "wind") v = Math.hypot(at(h, "u_blavg", iy, ix), at(h, "v_blavg", iy, ix));
          else if (k === "solar") v = ((utcH + lon / 15) % 24 + 24) % 24;
          else if (k === "doy_s") v = ds;
          else if (k === "doy_c") v = dc;
          else if (k === "relief") v = relief ? relief[iy * nx + ix] : NaN;
          else if (k === "model") v = mid;
          else if (k === "lat") v = lat;
          else v = at(h, k, iy, ix);
          x[feat[k]] = v;
        }
        const cb = at(h, "cubase", iy, ix), blue = !Number.isFinite(cb);
        if (F.ws) put(h, "wstar", iy, ix, ws * clamp(Math.exp(predict(F.ws, row(F.ws))), cl.ws));
        const key = blue && F.top ? "top" : F.zi ? "zi" : null;
        if (key) {
          const f = clamp(Math.exp(predict(F[key], row(F[key]))), cl[key]);
          for (const name of ["zi", "depth"]) if (fi[name] != null && Number.isFinite(at(h, name, iy, ix))) put(h, name, iy, ix, at(h, name, iy, ix) * f);
        }
        if (F.cb && !blue) put(h, "cubase", iy, ix, Math.max(50, cb + Math.min(cl.cb, Math.max(-cl.cb, predict(F.cb, row(F.cb))))));
        if (F.cu && fi.cubase != null) {
          const p = sigmoid(predict(F.cu, row(F.cu)));
          if (!blue && p < cl.cu[0]) {
            put(h, "cubase", iy, ix, NaN);
            if (fi.cudepth != null) put(h, "cudepth", iy, ix, NaN);
          } else if (blue && p > cl.cu[1]) {
            const zi = fi.zi != null ? out[(h * NF + fi.zi) * N + iy * nx + ix] : NaN;
            if (Number.isFinite(zi) && zi > 0) {
              put(h, "cubase", iy, ix, zi);
              if (fi.cudepth != null) put(h, "cudepth", iy, ix, cuDepth);
            }
          }
        }
      }
    }
  }
  return out;
}
export {
  CLAMPS,
  CU_DEPTH,
  RELIEF_DEG,
  applyCorrections,
  reliefGrid
};
