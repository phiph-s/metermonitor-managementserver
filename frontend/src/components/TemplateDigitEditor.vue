<template>
  <Teleport to="body" :disabled="!isFullscreen">
    <div class="editor-shell" :class="{ fullscreen: isFullscreen }">
      <div class="editor-panel">
        <button class="fullscreen-toggle" type="button" @click="toggleFullscreen">
          {{ isFullscreen ? 'Exit fullscreen' : 'Fullscreen' }}
        </button>
        <div class="template-editor">
          <div ref="canvas" class="template-canvas" :style="canvasStyle">
            <img :src="imageSrc" alt="Template" @load="onImageLoad" />
            <svg class="template-overlay" viewBox="0 0 1 1" preserveAspectRatio="none">
              <polygon
                v-for="(quad, index) in normalizedQuads"
                :key="`poly-${index}`"
                :points="quadPoints(quad)"
                class="digit-polygon"
                :class="{ active: index === activeIndex, inactive: index !== activeIndex }"
                @pointerdown.stop="setActive(index)"
              />
            </svg>
            <div
              v-for="(quad, index) in normalizedQuads"
              :key="`handle-${index}`"
              class="digit-handle"
              :class="{ active: index === activeIndex, inactive: index !== activeIndex }"
              :style="pointStyle(quadCenter(quad))"
              @pointerdown="startMove(index, $event)"
            >
              <span class="handle-label">{{ index + 1 }}</span>
            </div>
            <div
              v-for="(point, pointIndex) in activeQuad"
              v-if="activeQuad.length === 4"
              :key="`corner-${pointIndex}`"
              class="digit-corner"
              :style="pointStyle(point)"
              @pointerdown="startCornerDrag(activeIndex, pointIndex, $event)"
            />
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, watch, computed, nextTick } from 'vue';

const props = defineProps({
  imageSrc: { type: String, required: true },
  quads: { type: Array, default: () => [] },
  segments: { type: Number, default: 0 },
  displayCorners: { type: Array, default: () => [] }
});

const emit = defineEmits(['update:quads']);
const canvas = ref(null);
const containerSize = ref({ width: 0, height: 0 });
const normalizedQuads = ref([]);
const activeIndex = ref(0);
const dragging = ref(null);
const isFullscreen = ref(false);
const naturalSize = ref({ width: 0, height: 0 });
const viewportSize = ref({ width: window.innerWidth, height: window.innerHeight });
let resizeObserver = null;

const clamp = (value) => Math.max(0, Math.min(1, Number(value) || 0));

const normalizePoint = (point) => ({
  x: clamp(point.x ?? point[0]),
  y: clamp(point.y ?? point[1])
});

const normalizeQuad = (quad) => (quad || []).map((p) => normalizePoint(p)).slice(0, 4);

const defaultBounds = () => {
  if (Array.isArray(props.displayCorners) && props.displayCorners.length === 4) {
    const pts = props.displayCorners.map((p) => normalizePoint(p));
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    return {
      left: Math.min(...xs),
      right: Math.max(...xs),
      top: Math.min(...ys),
      bottom: Math.max(...ys)
    };
  }
  return { left: 0.1, right: 0.9, top: 0.3, bottom: 0.7 };
};

const buildDefaultQuads = (count) => {
  if (count < 1) return [];
  const bounds = defaultBounds();
  const left = clamp(bounds.left);
  const right = clamp(bounds.right);
  const top = clamp(bounds.top);
  const bottom = clamp(bounds.bottom);
  const width = Math.max(right - left, 0.05);
  const step = width / count;
  const quads = [];
  for (let i = 0; i < count; i += 1) {
    const x0 = left + step * i;
    const x1 = left + step * (i + 1);
    quads.push([
      { x: clamp(x0), y: top },
      { x: clamp(x1), y: top },
      { x: clamp(x1), y: bottom },
      { x: clamp(x0), y: bottom }
    ]);
  }
  return quads;
};

const mergeQuads = (incoming, count) => {
  if (count <= 0) return [];
  const normalized = Array.isArray(incoming) ? incoming.map((quad) => normalizeQuad(quad)) : [];
  if (normalized.length >= count) return normalized.slice(0, count);
  const defaults = buildDefaultQuads(count);
  const merged = normalized.slice();
  for (let i = normalized.length; i < count; i += 1) {
    merged.push(defaults[i] || defaults[defaults.length - 1]);
  }
  return merged;
};

const syncQuads = (incoming, count) => {
  const next = mergeQuads(incoming, count);
  const normalizedIncoming = Array.isArray(incoming) ? incoming.map((quad) => normalizeQuad(quad)) : [];
  const shouldEmit = JSON.stringify(normalizedIncoming) !== JSON.stringify(next);
  normalizedQuads.value = next;
  if (activeIndex.value >= next.length) activeIndex.value = 0;
  if (shouldEmit) {
    emit('update:quads', normalizedQuads.value);
  }
};

watch(
  () => [props.quads, props.segments, props.displayCorners],
  ([nextQuads, nextSegments]) => {
    syncQuads(nextQuads, Number(nextSegments) || 0);
  },
  { immediate: true, deep: true }
);

const activeQuad = computed(() => normalizedQuads.value[activeIndex.value] || []);

const quadPoints = (quad) => quad.map((p) => `${p.x},${p.y}`).join(' ');

const quadCenter = (quad) => {
  if (!quad || quad.length === 0) return { x: 0.5, y: 0.5 };
  const total = quad.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
  return { x: total.x / quad.length, y: total.y / quad.length };
};

const pointStyle = (point) => ({
  left: `${point.x * 100}%`,
  top: `${point.y * 100}%`,
  transform: 'translate(-50%, -50%)'
});

const onImageLoad = () => {
  if (canvas.value) {
    const img = canvas.value.querySelector('img');
    if (img) {
      naturalSize.value = { width: img.naturalWidth || 0, height: img.naturalHeight || 0 };
    }
  }
  updateContainerSize();
};

const updateContainerSize = () => {
  if (!canvas.value) return;
  const rect = canvas.value.getBoundingClientRect();
  containerSize.value = { width: rect.width, height: rect.height };
  viewportSize.value = { width: window.innerWidth, height: window.innerHeight };
};

const getNormalizedPointer = (event) => {
  if (!canvas.value) return { x: 0, y: 0 };
  const rect = canvas.value.getBoundingClientRect();
  const x = clamp((event.clientX - rect.left) / rect.width);
  const y = clamp((event.clientY - rect.top) / rect.height);
  return { x, y };
};

const startMove = (index, event) => {
  activeIndex.value = index;
  const pos = getNormalizedPointer(event);
  dragging.value = { type: 'move', quadIndex: index, last: pos };
  event.preventDefault();
  window.addEventListener('pointermove', onDrag);
  window.addEventListener('pointerup', stopDrag);
};

const startCornerDrag = (quadIndex, cornerIndex, event) => {
  activeIndex.value = quadIndex;
  const pos = getNormalizedPointer(event);
  dragging.value = { type: 'corner', quadIndex, cornerIndex, last: pos };
  event.preventDefault();
  window.addEventListener('pointermove', onDrag);
  window.addEventListener('pointerup', stopDrag);
};

const onDrag = (event) => {
  if (!dragging.value) return;
  const pos = getNormalizedPointer(event);
  const { type, quadIndex, cornerIndex, last } = dragging.value;
  const quads = normalizedQuads.value.map((quad) => quad.map((p) => ({ ...p })));
  if (!quads[quadIndex]) return;
  if (type === 'corner') {
    quads[quadIndex][cornerIndex] = { x: pos.x, y: pos.y };
  } else if (type === 'move') {
    const dx = pos.x - last.x;
    const dy = pos.y - last.y;
    quads[quadIndex] = quads[quadIndex].map((p) => ({
      x: clamp(p.x + dx),
      y: clamp(p.y + dy)
    }));
    dragging.value.last = pos;
  }
  normalizedQuads.value = quads;
  emit('update:quads', normalizedQuads.value);
};

const stopDrag = () => {
  dragging.value = null;
  window.removeEventListener('pointermove', onDrag);
  window.removeEventListener('pointerup', stopDrag);
};

const toggleFullscreen = async () => {
  isFullscreen.value = !isFullscreen.value;
  await nextTick();
  updateContainerSize();
};

const canvasStyle = computed(() => {
  if (!isFullscreen.value) return {};
  const maxWidth = Math.max(viewportSize.value.width - 200, 0);
  const maxHeight = Math.max(viewportSize.value.height - 200, 0);
  const nw = naturalSize.value.width || 0;
  const nh = naturalSize.value.height || 0;
  if (!nw || !nh || !maxWidth || !maxHeight) {
    return { width: `${maxWidth}px`, height: `${maxHeight}px` };
  }
  const scale = Math.min(maxWidth / nw, maxHeight / nh);
  const width = Math.max(1, Math.round(nw * scale));
  const height = Math.max(1, Math.round(nh * scale));
  return { width: `${width}px`, height: `${height}px` };
});

onMounted(() => {
  updateContainerSize();
  if (window.ResizeObserver) {
    resizeObserver = new ResizeObserver(updateContainerSize);
    if (canvas.value) {
      resizeObserver.observe(canvas.value);
    }
  }
  window.addEventListener('resize', updateContainerSize);
});

onBeforeUnmount(() => {
  stopDrag();
  if (resizeObserver && canvas.value) {
    resizeObserver.unobserve(canvas.value);
  }
  window.removeEventListener('resize', updateContainerSize);
});
</script>

<style scoped>
.template-editor {
  position: relative;
  width: 100%;
}

.template-canvas {
  position: relative;
  width: 100%;
  display: inline-block;
}

.template-canvas img {
  width: 100%;
  display: block;
}

.editor-shell {
  position: relative;
}

.editor-shell.fullscreen {
  position: fixed;
  inset: 0;
  z-index: 3000;
  background: rgba(0, 0, 0, 0.8);
  padding: 24px;
  box-sizing: border-box;
  display: flex;
}

.editor-panel {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.editor-shell.fullscreen .template-editor {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.editor-shell.fullscreen .template-canvas {
  width: auto;
  height: auto;
}

.editor-shell.fullscreen .template-canvas img {
  width: 100%;
  height: 100%;
}

.fullscreen-toggle {
  position: absolute;
  top: 12px;
  right: 12px;
  z-index: 5;
  background: rgba(20, 20, 20, 0.85);
  color: #fff;
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
}

.fullscreen-toggle:hover {
  background: rgba(20, 20, 20, 0.95);
}

.template-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: auto;
}

.digit-polygon {
  fill: rgba(190, 190, 190, 0.12);
  stroke: rgba(200, 200, 200, 0.55);
  stroke-width: 0.004;
  cursor: pointer;
}

.digit-polygon.inactive {
  fill: rgba(190, 190, 190, 0.12);
  stroke: rgba(200, 200, 200, 0.55);
}

.digit-polygon.active {
  fill: rgba(255, 80, 80, 0.18);
  stroke: rgba(255, 80, 80, 0.95);
}

.digit-handle {
  position: absolute;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: rgba(30, 30, 30, 0.8);
  border: 2px solid rgba(255, 255, 255, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: grab;
  z-index: 2;
}

.digit-handle.inactive {
  background: rgba(60, 60, 60, 0.7);
  border-color: rgba(255, 255, 255, 0.7);
}

.digit-handle.active {
  background: rgba(220, 60, 60, 0.9);
  border-color: rgba(255, 255, 255, 0.95);
}

.handle-label {
  font-size: 8px;
  color: #fff;
  font-weight: 700;
}

.digit-corner {
  position: absolute;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.9);
  border: 2px solid rgba(30, 30, 30, 0.9);
  cursor: grab;
  z-index: 3;
}
</style>
