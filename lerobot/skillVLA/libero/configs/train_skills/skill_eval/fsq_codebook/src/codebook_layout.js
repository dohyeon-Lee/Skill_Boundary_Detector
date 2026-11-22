// Keep axes 1–3 inside a cube; each combination of later axes gets its own panel.
// Shared by replay, full-training counts, comparisons, and diagnostic reports.
function fsqCubeGrid(levels) {
  const slices = levels.slice(3).reduce((a, b) => a * b, 1);
  const columns = Math.min(3, slices), width = 600 / columns;
  const height = (levels.length > 3 ? 580 : 520) / columns;
  return {slices, columns, width, height, rows: Math.ceil(slices / columns)};
}

function fsqProject(c, levels) {
  const grid = fsqCubeGrid(levels), visible = levels.slice(0, 3);
  const den = Math.max(1, Math.max(...visible) - 1) / 2;
  const v = visible.map((level, i) => ((c[i] || 0) - (level - 1) / 2) / den);
  let x, y, z;
  if (visible.length < 3) {
    x = 300 + (v[0] || 0) * 190;
    y = 265 - (v[1] || 0) * 190;
    z = 0;
  } else {
    const yaw = -.63, pitch = .46, cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const xr = cy * v[0] - sy * v[1], yr = sy * v[0] + cy * v[1];
    x = 300 + xr * 145;
    y = 265 + yr * 145 * sp - v[2] * 145 * cp;
    z = yr * cp + v[2] * sp;
  }
  let slice = 0, stride = 1;
  for (let d = 3; d < levels.length; d++) {
    slice += c[d] * stride;
    stride *= levels[d];
  }
  const scale = grid.width / 600, title = levels.length > 3 ? 25 * scale : 0;
  return [(slice % grid.columns) * grid.width + x * scale,
          Math.floor(slice / grid.columns) * grid.height + y * scale + title, z];
}

function fsqPrepareCube(svg, levels) {
  const grid = fsqCubeGrid(levels), NS = 'http://www.w3.org/2000/svg';
  svg.setAttribute('viewBox', `0 0 600 ${grid.rows * grid.height}`);
  svg.setAttribute('data-slice-count', grid.slices);
  if (levels.length <= 3) return;
  for (let slice = 0; slice < grid.slices; slice++) {
    let index = slice;
    const labels = [];
    for (let d = 3; d < levels.length; d++) {
      const bin = index % levels[d];
      index = Math.floor(index / levels[d]);
      // Odd FSQ levels have symmetric integer coordinates; even grids use bin IDs.
      const value = levels[d] % 2 ? bin - Math.floor(levels[d] / 2) : `bin ${bin}`;
      labels.push(`axis ${d + 1} = ${value}`);
    }
    const x = (slice % grid.columns) * grid.width;
    const y = Math.floor(slice / grid.columns) * grid.height;
    const panel = document.createElementNS(NS, 'rect');
    for (const [key, value] of Object.entries({x: x + 2, y: y + 2,
      width: grid.width - 4, height: grid.height - 4, rx: 6,
      fill: 'none', stroke: '#e0e5ed', 'pointer-events': 'none'})) panel.setAttribute(key, value);
    svg.appendChild(panel);
    const label = document.createElementNS(NS, 'text');
    for (const [key, value] of Object.entries({x: x + grid.width / 2, y: y + 17,
      'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: '#475467'})) label.setAttribute(key, value);
    label.textContent = labels.join(' · ');
    svg.appendChild(label);
  }
}

function fsqSamePosition(token, selected, levels) {
  if (selected < 0 || levels.length <= 3 || token === selected) return false;
  const size = levels.slice(0, 3).reduce((a, b) => a * b, 1);
  return token % size === selected % size;
}
