function httpError(status, message) {
  const error = new Error(message);
  error.status = status;
  return error;
}

function errorHandler(error, req, res, next) {
  if (res.headersSent) {
    next(error);
    return;
  }

  if (error.type === "entity.parse.failed") {
    res.status(400).json({ error: "Request body is not valid JSON." });
    return;
  }

  const status = Number.isInteger(error.status) ? error.status : 500;

  if (status >= 500) {
    console.error(error);
  }

  res.status(status).json({
    error: error.message || "Internal server error."
  });
}

module.exports = {
  httpError,
  errorHandler
};
