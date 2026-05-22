export const METER_TYPES = ['WATER', 'GAS', 'ELECTRICITY', 'CUSTOM'];

export const meterTypeColors = {
  WATER: '#3b82f6',
  GAS: '#9ca3af',
  ELECTRICITY: '#f59e0b',
  CUSTOM: '#6b7280',
};

export const meterTypeLabels = {
  WATER: 'Water',
  GAS: 'Gas',
  ELECTRICITY: 'Electricity',
  CUSTOM: 'Custom',
};

export const meterTypeDefaultUnits = {
  WATER: 'm³',
  GAS: 'm³',
  ELECTRICITY: 'kWh',
  CUSTOM: null,
};

export function getMeterUnit(meterType, customUnit) {
  if (meterType === 'CUSTOM') {
    return customUnit || '—';
  }
  return meterTypeDefaultUnits[meterType] || 'm³';
}

export function getMeterTypeColor(meterType) {
  return meterTypeColors[meterType] || meterTypeColors.CUSTOM;
}

export function getMeterTypeLabel(meterType) {
  return meterTypeLabels[meterType] || meterType || 'Water';
}
