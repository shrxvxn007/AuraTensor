#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AuraTensor minimal Maven Wrapper (single-file, no maven-wrapper.jar dep).
# Locates a system mvn first; otherwise downloads Apache Maven 3.9.6 into
# ~/.m2/wrapper and reuses it.

set -euf -o pipefail

MAVEN_VERSION="3.9.6"
WRAPPER_DIR="${HOME}/.m2/wrapper/dists/apache-maven-${MAVEN_VERSION}"
MVN_BIN="${WRAPPER_DIR}/apache-maven-${MAVEN_VERSION}/bin/mvn"

# 1) Prefer a system mvn
if command -v mvn >/dev/null 2>&1; then
  exec mvn "$@"
fi

# 2) Reuse cached install
if [ -x "${MVN_BIN}" ]; then
  exec "${MVN_BIN}" "$@"
fi

# 3) Bootstrap Maven
echo "[mvnw] Apache Maven not found on PATH. Downloading Maven ${MAVEN_VERSION}..."
mkdir -p "${WRAPPER_DIR}"

# Detect platform
case "$(uname -s)" in
  Linux*)   OS=linux ;;
  Darwin*)  OS=mac ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows ;;
  *)        echo "[mvnw] Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

case "${OS}" in
  linux|mac) ARCHIVE="apache-maven-${MAVEN_VERSION}-bin.tar.gz" ;;
  windows)   ARCHIVE="apache-maven-${MAVEN_VERSION}-bin.zip" ;;
esac

URL="https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/${ARCHIVE}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 -o "${TMP}/${ARCHIVE}" "${URL}"
else
  wget -q -O "${TMP}/${ARCHIVE}" "${URL}"
fi

if [ "${OS}" = "windows" ]; then
  unzip -q "${TMP}/${ARCHIVE}" -d "${WRAPPER_DIR}"
else
  tar -xzf "${TMP}/${ARCHIVE}" -C "${WRAPPER_DIR}"
fi

echo "[mvnw] Maven ${MAVEN_VERSION} installed to ${WRAPPER_DIR}"
exec "${MVN_BIN}" "$@"
