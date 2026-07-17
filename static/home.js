const heroCanvas = document.querySelector("#heroCanvas");
const heroContext = heroCanvas.getContext("2d");

function resizeHeroCanvas() {
  const rect = heroCanvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  heroCanvas.width = Math.max(1, Math.floor(rect.width * ratio));
  heroCanvas.height = Math.max(1, Math.floor(rect.height * ratio));
  heroContext.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function roundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawMonitorCard(ctx, x, y, width, height, phase) {
  roundedRect(ctx, x, y, width, height, 8);
  ctx.fillStyle = "rgba(255,255,255,0.1)";
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.72)";
  ctx.font = "700 13px Inter, sans-serif";
  ctx.fillText("Live Model Snapshot", x + 22, y + 32);

  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 6; i += 1) {
    const gy = y + 50 + i * ((height - 72) / 6);
    ctx.beginPath();
    ctx.moveTo(x + 22, gy);
    ctx.lineTo(x + width - 22, gy);
    ctx.stroke();
  }

  ctx.strokeStyle = "#7ae0c7";
  ctx.lineWidth = 3;
  ctx.beginPath();
  const startX = x + 22;
  const endX = x + width - 22;
  const midY = y + height * 0.58;
  for (let px = startX; px <= endX; px += 4) {
    const cycle = (px + phase * 70) % 132;
    let py = midY + Math.sin((px + phase * 120) / 18) * 4;
    if (cycle > 18 && cycle < 31) py -= (cycle - 18) * 2.4;
    if (cycle >= 31 && cycle < 42) py += (cycle - 31) * 5 - 30;
    if (cycle >= 42 && cycle < 58) py -= (cycle - 42) * 1.6 - 26;
    if (cycle >= 82 && cycle < 112) py -= Math.sin(((cycle - 82) / 30) * Math.PI) * 13;
    if (px === startX) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  const gaugeX = x + width - 96;
  const gaugeY = y + 42;
  ctx.beginPath();
  ctx.arc(gaugeX, gaugeY, 34, -Math.PI / 2, Math.PI * 1.15);
  ctx.strokeStyle = "rgba(255,255,255,0.14)";
  ctx.lineWidth = 9;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(gaugeX, gaugeY, 34, -Math.PI / 2, Math.PI * 0.72);
  ctx.strokeStyle = "#d99721";
  ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.font = "800 18px Inter, sans-serif";
  ctx.fillText("70%", gaugeX - 18, gaugeY + 7);
}

function drawHero(time = 0) {
  const rect = heroCanvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const phase = time / 1000;

  heroContext.clearRect(0, 0, width, height);
  heroContext.fillStyle = "#101923";
  heroContext.fillRect(0, 0, width, height);

  for (let y = 0; y < height; y += 34) {
    heroContext.strokeStyle = "rgba(255,255,255,0.055)";
    heroContext.lineWidth = 1;
    heroContext.beginPath();
    heroContext.moveTo(0, y);
    heroContext.lineTo(width, y);
    heroContext.stroke();
  }

  for (let x = 0; x < width; x += 34) {
    heroContext.strokeStyle = "rgba(255,255,255,0.04)";
    heroContext.beginPath();
    heroContext.moveTo(x, 0);
    heroContext.lineTo(x, height);
    heroContext.stroke();
  }

  if (width >= 760) {
    const cardWidth = Math.min(560, width * 0.42);
    const cardHeight = 260;
    drawMonitorCard(
      heroContext,
      Math.max(width * 0.54, width - cardWidth - 70),
      Math.max(110, height * 0.2),
      cardWidth,
      cardHeight,
      phase,
    );
  }

  heroContext.fillStyle = "rgba(216,79,53,0.82)";
  heroContext.beginPath();
  heroContext.arc(width * 0.82, height * 0.68, 8 + Math.sin(phase * 2) * 2, 0, Math.PI * 2);
  heroContext.fill();

  heroContext.fillStyle = "rgba(16,131,111,0.72)";
  heroContext.beginPath();
  heroContext.arc(width * 0.74, height * 0.78, 6 + Math.cos(phase * 1.6) * 2, 0, Math.PI * 2);
  heroContext.fill();

  requestAnimationFrame(drawHero);
}

resizeHeroCanvas();
window.addEventListener("resize", resizeHeroCanvas);
requestAnimationFrame(drawHero);
