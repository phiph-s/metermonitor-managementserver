export const SOURCE_COLORS = {
  mqtt: '#a855f7',
  ha_camera: '#41bdf5',
  http: '#9ca3af',
  unknown: '#94a3b8'
};

export const SOURCE_LABELS = {
  mqtt: 'MQTT',
  ha_camera: 'Home Assistant',
  http: 'HTTP',
  unknown: 'Unknown'
};

export const normalizeSourceType = (sourceType) => {
  if (!sourceType) return 'unknown';
  const normalized = String(sourceType).toLowerCase();
  if (normalized === 'ha' || normalized === 'homeassistant' || normalized === 'camera') {
    return 'ha_camera';
  }
  if (normalized === 'http' || normalized === 'http_endpoint' || normalized === 'webhook') {
    return 'http';
  }
  if (normalized === 'mqtt') {
    return 'mqtt';
  }
  return normalized;
};

export const getSourceColor = (sourceType) => {
  const normalized = normalizeSourceType(sourceType);
  return SOURCE_COLORS[normalized] || SOURCE_COLORS.unknown;
};

export const getSourceLabel = (sourceType) => {
  const normalized = normalizeSourceType(sourceType);
  return SOURCE_LABELS[normalized] || SOURCE_LABELS.unknown;
};
