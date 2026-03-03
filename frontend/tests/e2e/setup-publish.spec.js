const { test, expect } = require('@playwright/test');
const mqtt = require('mqtt');
const path = require('path');
const { readFileSync } = require('fs');

const MQTT_URL = process.env.E2E_MQTT_URL || 'mqtt://127.0.0.1:1889';
const MQTT_PUB_TOPIC = process.env.E2E_MQTT_PUB_TOPIC || 'MeterMonitor/test';
const HA_IMAGE_URL = process.env.E2E_HTTP_IMAGE_URL || 'http://127.0.0.1:1888/test-image.png';

const repoRoot = path.resolve(__dirname, '..', '..', '..');
const imagePath = path.join(repoRoot, 'test', 'img', 'img.png');
const imageBase64 = readFileSync(imagePath).toString('base64');

const connectOnce = (timeoutMs = 3000) =>
  new Promise((resolve, reject) => {
    const client = mqtt.connect(MQTT_URL, {
      connectTimeout: timeoutMs,
      reconnectPeriod: 0,
    });
    const timer = setTimeout(() => {
      client.end(true);
      reject(new Error(`MQTT connect timeout after ${timeoutMs}ms`));
    }, timeoutMs);
    client.on('connect', () => {
      clearTimeout(timer);
      resolve(client);
    });
    client.on('error', (err) => {
      clearTimeout(timer);
      client.end(true);
      reject(err);
    });
  });

const connectMqtt = async (totalTimeoutMs = 15000) => {
  const start = Date.now();
  let lastError;
  while (Date.now() - start < totalTimeoutMs) {
    try {
      return await connectOnce(3000);
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError || new Error('MQTT connect failed');
};

const publishImage = (client, name, pictureNumber, timestamp) =>
  new Promise((resolve, reject) => {
    const payload = {
      name,
      picture_number: pictureNumber,
      'WiFi-RSSI': -40,
      picture: {
        timestamp,
        format: 'png',
        width: 640,
        height: 480,
        length: imageBase64.length,
        data: imageBase64,
      },
    };

    client.publish(MQTT_PUB_TOPIC, JSON.stringify(payload), { qos: 1 }, (err) => {
      if (err) reject(err);
      else resolve();
    });
  });

const waitForMqttMessage = (client, topic, timeoutMs = 120000) =>
  new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error(`Timeout waiting for MQTT message on ${topic}`));
    }, timeoutMs);

    const handler = (msgTopic, message) => {
      if (msgTopic === topic) {
        clearTimeout(timeout);
        client.off('message', handler);
        resolve(message);
      }
    };

    client.subscribe(topic, { qos: 1 }, (err) => {
      if (err) {
        clearTimeout(timeout);
        reject(err);
        return;
      }
      client.on('message', handler);
    });
  });

const openAddSourceDialog = async (page) => {
  const addCard = page.getByTestId('add-watermeter-card').first();
  await addCard.click();
  await expect(page.getByTestId('source-type-select')).toBeVisible({ timeout: 30000 });
};

const selectSourceType = async (page, label) => {
  await page.getByTestId('source-type-select').click();
  const option = page.locator('.n-base-select-option', { hasText: label }).first();
  await expect(option).toBeVisible({ timeout: 30000 });
  await option.click();
};

const createHttpSource = async (page, name) => {
  await openAddSourceDialog(page);
  await selectSourceType(page, 'HTTP (URL)');
  await expect(page.getByTestId('http-url-input')).toBeVisible({ timeout: 30000 });
  await page.getByTestId('source-name-input').locator('input').fill(name);
  await page.getByTestId('source-poll-interval').locator('input').fill('10');
  await page.getByTestId('http-url-input').locator('input').fill(HA_IMAGE_URL);
  const createButton = page.getByTestId('create-source-button');
  await expect(createButton).toBeEnabled({ timeout: 30000 });
  const [response] = await Promise.all([
    page.waitForResponse(
      (res) => res.url().includes('/api/sources') && res.request().method() === 'POST'
    ),
    createButton.click(),
  ]);
  if (!response.ok()) {
    const body = await response.text();
    throw new Error(`Create HTTP source failed: ${response.status()} ${body}`);
  }
  await expect(page.getByTestId('create-source-button')).toBeHidden({ timeout: 30000 });
};

const createHaSource = async (page, name) => {
  await openAddSourceDialog(page);
  await selectSourceType(page, 'Home Assistant (Camera entity)');
  await page.getByTestId('source-name-input').locator('input').fill(name);
  await page.getByTestId('source-poll-interval').locator('input').fill('10');
  await page.getByTestId('ha-camera-select').click();
  const camOption = page.locator('.n-base-select-option', { hasText: /Test Camera/i }).first();
  await expect(camOption).toBeVisible({ timeout: 30000 });
  await camOption.click();
  const createButton = page.getByTestId('create-source-button');
  await expect(createButton).toBeEnabled({ timeout: 30000 });
  const [response] = await Promise.all([
    page.waitForResponse(
      (res) => res.url().includes('/api/sources') && res.request().method() === 'POST'
    ),
    createButton.click(),
  ]);
  if (!response.ok()) {
    const body = await response.text();
    throw new Error(`Create HA source failed: ${response.status()} ${body}`);
  }
  await expect(page.getByTestId('create-source-button')).toBeHidden({ timeout: 30000 });
};

const expectMeterCardVisible = async (page, name) => {
  const title = page.locator('.meter-card .card-title', { hasText: name }).first();
  await expect(title).toBeVisible({ timeout: 120000 });
};

const setupMeter = async (page, name) => {
  await expectMeterCardVisible(page, name);
  const card = page.locator('.meter-card', { hasText: name });
  await card.getByRole('button', { name: 'Setup' }).click();

  await expect(page.getByText(`Setup for ${name}`)).toBeVisible({ timeout: 60000 });

  const segmentsInput = page
    .getByText('Segments')
    .locator('..')
    .locator('.n-input__input-el')
    .first();
  await segmentsInput.fill('7');

  const rotatedRow = page.locator('.n-flex', { hasText: '180° rotated' }).first();
  const rotatedSwitch = rotatedRow.locator('[role="switch"]').first();
  if ((await rotatedSwitch.getAttribute('aria-checked')) !== 'true') {
    await rotatedSwitch.click();
  }

  const nextButton = page.getByRole('button', { name: 'Next' });
  await expect(nextButton).toBeEnabled({ timeout: 180000 });
  await nextButton.click();

  const applyButton = page.getByRole('button', { name: 'Apply' });
  await expect(applyButton).toBeEnabled({ timeout: 180000 });
  await expect(page.getByText('0 - 125')).toHaveCount(2);
  await applyButton.click();

  const readoutInput = page.getByPlaceholder('Readout');
  await readoutInput.click();
  await readoutInput.fill('1');

  const maxFlowInput = page
    .locator('text=Max. flow rate')
    .locator('..')
    .locator('.n-input__input-el')
    .first();
  await maxFlowInput.fill('10000');

  const finishButton = page.getByRole('button', { name: 'Finish & save' });
  await expect(finishButton).toBeEnabled({ timeout: 180000 });
  await finishButton.click();

  await expect(page).toHaveURL(new RegExp(`#\\/meter\\/${name}$`), { timeout: 60000 });
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('secret', 'test_token');
  });
});

test('sources setup and mqtt publish', async ({ page }) => {
  const mqttName = 'meter-mqtt';
  const httpName = 'meter-http';
  const haName = 'meter-ha';

  await page.goto('/#/', { waitUntil: 'domcontentloaded' });
  await expect(page.getByTestId('add-watermeter-card').first()).toBeVisible({ timeout: 60000 });

  const client = await connectMqtt();

  try {
    await publishImage(client, mqttName, 1, '2025-01-01T00:00:00');
    await expect.poll(async () => {
      const res = await page.request.get('/api/discovery', {
        headers: { secret: 'test_token' },
      });
      if (!res.ok()) return false;
      const data = await res.json();
      return (data.watermeters || []).some((row) => row[0] === mqttName);
    }, { timeout: 120000 }).toBeTruthy();

    await page.reload({ waitUntil: 'domcontentloaded' });
    await expectMeterCardVisible(page, mqttName);

    await createHttpSource(page, httpName);
    await expectMeterCardVisible(page, httpName);

    await createHaSource(page, haName);
    await expectMeterCardVisible(page, haName);

    await setupMeter(page, mqttName);
    await page.goto('/#/');

    await setupMeter(page, httpName);
    await page.goto('/#/');

    await setupMeter(page, haName);
    await page.goto('/#/');

    const valueTopic = `MeterMonitor/${mqttName}/value`;
    const valuePromise = waitForMqttMessage(client, valueTopic, 120000);

    await publishImage(client, mqttName, 2, '2025-01-01T00:10:00');

    const message = await valuePromise;
    const payload = JSON.parse(message.toString());
    expect(payload.value).toBeGreaterThan(0);
  } finally {
    client.end(true);
  }
});
