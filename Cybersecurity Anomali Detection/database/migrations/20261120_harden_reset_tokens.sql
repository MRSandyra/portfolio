-- Backward-compatible migration for existing installations.
-- New values in reset_token are SHA-256 hashes; existing plaintext tokens are invalidated.
ALTER TABLE login
    MODIFY reset_token VARCHAR(64) NULL,
    MODIFY reset_token_expiry DATETIME NULL;

UPDATE login
SET reset_token = NULL, reset_token_expiry = NULL
WHERE reset_token IS NOT NULL;

-- Run once. This index makes reset-token lookup efficient without changing application columns.
CREATE INDEX idx_login_reset_token ON login (reset_token);
