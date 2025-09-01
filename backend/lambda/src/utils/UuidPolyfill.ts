import { v4 } from 'uuid';

/**
 * UUID v6 polyfill for older UUID versions that don't support v6
 * This implements a basic UUID v6 generator using v4 as a fallback
 */
export function v6(options?: { clockseq?: number }): string {
  // For now, use v4 as a fallback since v6 is not available in older UUID versions
  // UUID v6 is a time-ordered version of UUID v1, but for our use case, v4 is sufficient
  return v4();
}

/**
 * Generate a simple UUID-like string as a fallback
 */
export function generateSimpleUuid(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Monkey patch the uuid module to add v6 support if it doesn't exist
 */
export function patchUuidModule(): void {
  try {
    // Try to require the uuid module
    const uuid = require('uuid');
    
    // Only patch if v6 doesn't exist
    if (!uuid.v6) {
      uuid.v6 = v6;
      console.log('UUID v6 polyfill applied successfully');
    } else {
      console.log('UUID v6 already available, no polyfill needed');
    }
  } catch (error) {
    console.warn('Failed to patch UUID module, creating global fallback:', error);
    
    // Create a global fallback if the module is not available
    try {
      (global as any).uuid = {
        v6: v6,
        v4: v4
      };
      console.log('Global UUID fallback created');
    } catch (fallbackError) {
      console.error('Failed to create UUID fallback:', fallbackError);
    }
  }
}

/**
 * Initialize the UUID polyfill
 * Call this at the start of your Lambda function
 */
export function initializeUuidPolyfill(): void {
  console.log('Initializing UUID polyfill...');
  patchUuidModule();
  console.log('UUID polyfill initialization complete');
}
