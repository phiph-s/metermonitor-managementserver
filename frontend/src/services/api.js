import { useAuthStore } from '@/stores/authStore';

const host = import.meta.env.VITE_HOST;

/**
 * Centralized API service with error handling and auth
 */
class ApiService {
  async request(url, options = {}) {
    const authStore = useAuthStore();
    
    const defaultOptions = {
      headers: {
        'secret': authStore.secret,
        ...options.headers,
      },
    };

    const response = await fetch(host + url, { ...defaultOptions, ...options });
    
    // Check auth status
    authStore.checkAuth(response);
    
    return response;
  }

  async get(url) {
    return this.request(url, { method: 'GET' });
  }

  async post(url, data) {
    const authStore = useAuthStore();
    return this.request(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'secret': authStore.secret,
      },
      body: JSON.stringify(data),
    });
  }

  async put(url, data) {
    const authStore = useAuthStore();
    return this.request(url, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'secret': authStore.secret,
      },
      body: JSON.stringify(data),
    });
  }

  async delete(url) {
    return this.request(url, { method: 'DELETE' });
  }

  async getJson(url) {
    const response = await this.get(url);
    if (response.ok) {
      return await response.json();
    }
    throw new Error(`API request failed: ${response.status}`);
  }

  async postJson(url, data) {
    const response = await this.post(url, data);
    if (response.ok) {
      return await response.json();
    }
    else{
      // extract error message from response
      const errorData = await response.json();
      throw new Error(errorData.detail || `API request failed: ${response.status}`);
    }
  }

  async putJson(url, data) {
    const response = await this.put(url, data);
    if (response.ok) {
      return await response.json();
    } else {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API request failed: ${response.status}`);
    }
  }
}

export const apiService = new ApiService();

