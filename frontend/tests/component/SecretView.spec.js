import { mount } from '@vue/test-utils';
import { vi } from 'vitest';

vi.mock('@/router', () => ({
  default: {
    push: vi.fn(),
  },
}));

import router from '@/router';
import SecretView from '@/views/SecretView.vue';

test('stores secret and navigates to root', async () => {
  const wrapper = mount(SecretView);
  const input = wrapper.find('input');
  const button = wrapper.find('button');

  await input.setValue('test_token');
  await button.trigger('click');

  expect(localStorage.getItem('secret')).toBe('test_token');
  expect(router.push).toHaveBeenCalledWith('/');
});
