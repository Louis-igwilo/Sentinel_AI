function initThemeToggle() {
  const html = document.documentElement;
  const currentTheme = localStorage.getItem("theme") || "dark";
  html.setAttribute("data-theme", currentTheme);

  let themeBtn = document.getElementById("theme-toggle");
  if (!themeBtn) {
    const navRight = document.querySelector(".nav-right");
    if (!navRight) return;
    themeBtn = document.createElement("button");
    themeBtn.id = "theme-toggle";
    themeBtn.className = "theme-toggle";
    themeBtn.type = "button";
    themeBtn.title = "Toggle light/dark mode";
    navRight.insertBefore(themeBtn, navRight.firstChild);
  }

  themeBtn.textContent = currentTheme === "dark" ? "☀" : "🌙";
  themeBtn.addEventListener("click", () => {
    const newTheme = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
    themeBtn.textContent = newTheme === "dark" ? "☀" : "🌙";
  });
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", initThemeToggle);
} else {
  initThemeToggle();
}
