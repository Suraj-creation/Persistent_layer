"""Safe transport failures and correlation without logging learner text or secrets."""
import logging
import sqlite3
import uuid

from fastapi.responses import JSONResponse
from packages.learner_state_core.assessment.execution import REQUEST_ID

logger = logging.getLogger('learner.api')


def install_reliability(app):
    @app.middleware('http')
    async def correlate(request, call_next):
        request_id = uuid.uuid4().hex
        token = REQUEST_ID.set(request_id)
        try:
            try:
                response = await call_next(request)
            except Exception as error:
                database_error = isinstance(error, sqlite3.Error)
                logger.error('request_failed request_id=%s method=%s error_type=%s', request_id, request.method, type(error).__name__)
                response = JSONResponse(status_code=503 if database_error else 500, content={
                    'detail': 'Local storage is unavailable. Keep your explanation and reload saved state before retrying.' if database_error else 'The request could not be completed. Reload saved state before retrying.',
                    'request_id': request_id,
                })
            response.headers['X-Request-ID'] = request_id
            return response
        finally:
            REQUEST_ID.reset(token)
