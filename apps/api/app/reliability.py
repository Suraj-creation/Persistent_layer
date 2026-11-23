"""Safe transport failures and correlation without logging learner text or secrets."""
# Log request failures with safe correlation fields.
import logging
# Identify storage failures separately from other request exceptions.
import sqlite3
# Generate a correlation ID per HTTP request.
import uuid

# Return a predictable JSON shape to browser callers.
from fastapi.responses import JSONResponse
# Share request correlation with the assessment execution journal.
from packages.learner_state_core.assessment.execution import REQUEST_ID

# Use the API's dedicated logger for transport failures.
logger = logging.getLogger('learner.api')


# Install the HTTP reliability middleware on the app.
def install_reliability(app):
    # Wrap every HTTP request handled by FastAPI.
    @app.middleware('http')
    # Set its correlation ID, translate failures and return the response.
    async def correlate(request, call_next):
        # Give the handling attempt a unique public request reference.
        request_id = uuid.uuid4().hex
        # Set that ID for nested trace work and remember the previous context.
        token = REQUEST_ID.set(request_id)
        # Always reset request-local context when handling ends.
        try:
            # Keep normal handling inside an unexpected-exception boundary.
            try:
                # Await the actual route handler or the next middleware.
                response = await call_next(request)
            # Catch unexpected errors the route did not already translate.
            except Exception as error:
                # Identify an unavailable SQLite operation as a storage error.
                database_error = isinstance(error, sqlite3.Error)
                # Log the request ID, method and exception type for troubleshooting.
                logger.error('request_failed request_id=%s method=%s error_type=%s', request_id, request.method, type(error).__name__)
                # Use service-unavailable for SQLite failures and internal-error for other
                # exceptions.
                response = JSONResponse(status_code=503 if database_error else 500, content={
                    # Explain the recovery action using safe text for the appropriate failure.
                    'detail': 'Local storage is unavailable. Keep your explanation and reload saved state before retrying.' if database_error else 'The request could not be completed. Reload saved state before retrying.',
                    # Include the same request reference in the error body.
                    'request_id': request_id,
                })
            # Attach the correlation ID on both normal and failed responses.
            response.headers['X-Request-ID'] = request_id
            # Return the API response to the browser.
            return response
        # Clean up the request-specific context regardless of outcome.
        finally:
            # Restore the earlier context so the ID does not leak into unrelated work.
            REQUEST_ID.reset(token)
