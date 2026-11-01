function predict(m, x) {
  const v = Array.isArray(x) ? x : m.features.map((k) => x[k]);
  let s = m.base;
  for (const t of m.trees) {
    let i = 0;
    while (t[i].length > 1) {
      const [f, thr, ml, l, r] = t[i], a = v[f];
      i = a == null || Number.isNaN(a) ? ml ? l : r : a <= thr ? l : r;
    }
    s += t[i][0];
  }
  return s;
}
const factor = (m, x) => Math.exp(predict(m, x));
const probability = (m, x) => 1 / (1 + Math.exp(-predict(m, x)));
const value = (m, x) => m.kind === "logit" ? probability(m, x) : m.kind === "add" ? predict(m, x) : factor(m, x);
export {
  factor,
  predict,
  probability,
  value
};
