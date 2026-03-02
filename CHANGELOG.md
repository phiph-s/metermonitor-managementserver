v3.3.0 - 06.02.2026
- Added E2E tests and Python unit-tests
- Fixed static extractor

v3.2.0 - 02.02.2026
- Fixed non-MQTT sources (HTTP, Home Assistant Camera) not publishing meter values to MQTT broker
  - Added dedicated MQTT publisher client for polling/capture operations
  - HTTP and HA Camera sources now publish both discovery registration and meter values
  - MQTT client is injected into PollingHandler and HTTP server capture flows
- All source types (MQTT, HTTP, HA Camera) now have consistent MQTT publishing behavior
- Added new ROI Extractor: **Static Rect**
  - Simple fixed rectangle cropping without alignment or transformation
  - User defines 4 corner points, extractor crops that exact region
  - Useful for stable camera positions where YOLO is unreliable
  - Supports perspective transform for non-axis-aligned rectangles
  - Available in frontend setup wizard alongside YOLO, BYPASS, and ORB extractors

v3.1.1-beta - 28.01.2026
- Added configurable correctional algorithm with Full/Light modes
  - Full mode: Complete correction with positive flow check, max flow rate validation, and fallback handling
  - Light mode: Only replaces rotation class and low-confidence digits with last known values
- Added toggle switch in setup to select between Full/Light correctional algorithm
- Added "Reset Corr. Alg." context menu item to mark all evaluations as outdated for re-evaluation
- Added API endpoint `/api/watermeters/{name}/evaluations/mark-outdated` to trigger re-evaluation
- Display current correction algorithm mode (Full/Light) in meter details
- Max flow rate input is automatically disabled when Light mode is selected
- Fixed ROI template editor not applying point updates correctly (v-model binding issue)
- Fixed display_corners not being properly scaled to pixel coordinates when saving ROI templates
- Added `use_correctional_alg` setting to database schema with migration support

v3.0.0 - 27.01.2026
- Implemented concept of image sources: MQTT, Home Assistant Camera, HTTP
- Added template-based ROI extractor (ORBExtractor) for manual display region selection
- Added ROI extraction framework with BypassExtractor, YOLOExtractor, and ORBExtractor
- Added selectable ROI extractors in setup with template configuration UI
- Added template management: create, edit, and configure reference points for ORB matching
- Added HTTP source support with custom headers, body, and URL configuration
- Added polling service for Home Assistant camera sources with configurable intervals
- Added source management UI: create, edit, and delete sources (MQTT, HA Camera, HTTP)
- Revamped frontend layouts with unified header controls and improved responsiveness for meter and source views
- Added source validation for water meters and a collapsible source editor in meter details
- Enhanced meter charts with combined usage/confidence series and improved layout
- Added correction evaluation metadata (flow rate, deltas, rejection reasons, digit-change stats) with full API + UI exposure
- Added evaluation detail dialog with dedicated API endpoint and click-to-view in evaluation results
- Overhauled evaluation list UX: infinite scroll, relative timestamps, per-digit grouping, and clearer outdated separation
- Added bounding box validation and warnings for missing bounding boxes in meter details/cards
- Added `used_confidence` persistence in history/evaluations for more consistent confidence reporting
- Improved Home Assistant camera capture flow with better error messaging and flash light configuration UX
- Removed obsolete endpoints and streamlined source/setup UI
- Fixed a critical bug that caused the bounding box to be rotated 90-180 degrees in some cases
- Test capturing in source creation
- Fixed UnicodeDecodeError by base64 encoding binary image data in API responses
- Added sources table migration for database schema
- Improved error handling and logging in capture and polling operations
- Fixed `/api/watermeters/{name}/evals/count` routing to avoid eval_id parsing errors
- Reduced excessive debug logging in Home Assistant token handling and HTTP requests
- Various UI improvements and bug fixes
- Added shared Home Assistant authentication utilities with centralized token management
- Implemented supervisor token support with automatic fallback to manual tokens
- Added homeassistant_api permission for proper supervisor endpoint access
- Improved Home Assistant API error handling with context-aware 401 error messages
- Optimized Docker build process with layer caching and platform-specific builds
- Fixed configuration loading with deep merge for nested config options
- Ensured default values from ha_default_settings.json are properly applied
