window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams",
    macros: {
      R: "\\mathbb{R}",
      E: "\\mathbb{E}",
      KL: "D_{\\mathrm{KL}}",
      N: "\\mathcal{N}",
      L: "\\mathcal{L}",
      bx: "\\mathbf{x}",
      by: "\\mathbf{y}",
      bz: "\\mathbf{z}",
      bY: "\\mathbf{Y}",
      bW: "\\mathbf{W}",
      argmax: "\\operatorname*{arg\\,max}",
      argmin: "\\operatorname*{arg\\,min}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
