#!/usr/bin/env bash
# Download official ABC Sim task tarballs selected by ABC_sim_dataset_config.yaml.
#
# Usage:
#   ./download_ABC_sim.sh --list-tasks
#   ./download_ABC_sim.sh --dry-run
#   ./download_ABC_sim.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../../../../../../../.venv/bin/python"
[ -x "${PYTHON}" ] || PYTHON=python3

configure_ssl_cert_file() {
  local cert_path
  if [ -n "${SSL_CERT_FILE:-}" ]; then
    [ -f "${SSL_CERT_FILE}" ] || {
      echo "ERROR: SSL_CERT_FILE이 존재하지 않습니다: ${SSL_CERT_FILE}" >&2
      return 1
    }
    echo "[env] SSL CA bundle: ${SSL_CERT_FILE}"
    return 0
  fi

  for cert_path in \
    /etc/pki/tls/certs/ca-bundle.crt \
    /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/ssl/cert.pem
  do
    if [ -f "${cert_path}" ]; then
      export SSL_CERT_FILE="${cert_path}"
      echo "[env] SSL CA bundle detected: ${SSL_CERT_FILE}"
      return 0
    fi
  done

  cert_path="$("${PYTHON}" -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [ -n "${cert_path}" ] && [ -f "${cert_path}" ]; then
    export SSL_CERT_FILE="${cert_path}"
    echo "[env] SSL CA bundle from certifi: ${SSL_CERT_FILE}"
    return 0
  fi

  echo "WARNING: SSL CA bundle을 찾지 못했습니다. Python 기본 인증서 설정을 사용합니다." >&2
}

configure_ssl_cert_file
export ABC_SIM_ONLY="${ABC_SIM_ONLY:-}"
exec "${PYTHON}" "${SCRIPT_DIR}/src/download_abc_sim.py" "$@"
