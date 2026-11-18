document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea")) return;
  const link = document.querySelector(e.key === "ArrowRight" ? "a[rel=next]" : e.key === "ArrowLeft" ? "a[rel=prev]" : null);
  if (link) window.location.href = link.href;
});
