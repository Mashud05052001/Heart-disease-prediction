(function () {
  const key = "heart-risk-theme";
  const root = document.documentElement;
  const saved = localStorage.getItem(key);
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initialTheme = saved || (prefersDark ? "dark" : "light");

  function applyTheme(theme) {
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    document.querySelectorAll("[data-theme-toggle]").forEach((toggle) => {
      toggle.checked = theme === "dark";
    });
    window.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
  }

  applyTheme(initialTheme);

  window.addEventListener("DOMContentLoaded", () => {
    applyTheme(root.dataset.theme || initialTheme);
    document.querySelectorAll("[data-theme-toggle]").forEach((toggle) => {
      toggle.addEventListener("change", () => {
        const nextTheme = toggle.checked ? "dark" : "light";
        localStorage.setItem(key, nextTheme);
        applyTheme(nextTheme);
      });
    });
  });
})();
