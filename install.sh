#!/usr/bin/env sh

if ! tput sgr0 >/dev/null 2>&1; then
    export TERM=xterm
fi

set -eu  # exit on error, undefined vars are errors

# --- Colors & Emojis ---------------------------------------------------------
# Prettified logging helpers — if tput isn't available, fail silently.
BOLD="$(tput bold || true)"
RESET="$(tput sgr0 || true)"
GREEN="$(tput setaf 2 || true)"
YELLOW="$(tput setaf 3 || true)"
RED="$(tput setaf 1 || true)"
BLUE="$(tput setaf 4 || true)"

# Lightweight log helpers (color-coded, emoji-powered).
log_info()   { printf "💬 ${BLUE}%s${RESET}\n" "$1"; }
log_ok()     { printf "✅ ${GREEN}%s${RESET}\n" "$1"; }
log_warn()   { printf "⚠️  ${YELLOW}%s${RESET}\n" "$1"; }
log_error()  { printf "❌ ${RED}%s${RESET}\n" "$1" >&2; }

# --- Helper functions --------------------------------------------------------

# Detect OS for release asset naming.
get_os() {
    case "$(uname)" in
      Darwin) echo "apple-darwin" ;;
      Linux)  echo "unknown-linux-gnu" ;;
      *)
        log_error "Unsupported OS. Build from source instead."
        exit 1
        ;;
    esac
}

# Choose an install dir with write access, falling back sanely.
choose_install_dir() {
    if [ -w /usr/local/bin ]; then
        echo "/usr/local/bin"
        return
    fi

    # $HOME/.local/bin is usually safe, create if missing.
    if [ -d "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin"; then
        echo "$HOME/.local/bin"
        return
    fi

    # Worst-case fallback.
    echo "$HOME/bin"
}

# Detect CPU architecture for release asset naming.
get_arch() {
    arch="$(uname -m)"
    case "$arch" in
      x86_64) echo "x86_64" ;;
      arm64|aarch64) echo "aarch64" ;;
      *)
        log_error "Unknown architecture: $arch"
        exit 1
        ;;
    esac
}

# Main installer logic.
install_encoderfile() {
    version=$1
    bin_dir=$2

    # Resolve GitHub release path.
    if [ $version = "latest" ]; then
        release="latest"
    else
        release="tags/v$1"
    fi

    url="https://api.github.com/repos/mozilla-ai/encoderfile/releases/${release}"

    # --- detect platform ----------------------------------------------------

    log_info "Detecting platform…"
    os=$(get_os)
    arch=$(get_arch)
    log_ok "Platform: ${arch} / ${os}"

    asset_name="encoderfile-${arch}-${os}.tar.gz"

    # --- fetch metadata and locate asset -----------------------------------

    log_info "Fetching release metadata for v${version}…"
    response=$(curl -sSL "$url") || {
        log_error "Failed to contact GitHub."
        exit 1
    }

    # Extract download URL for the matching asset.
    asset_url=$(printf "%s" "$response" \
        | jq -er ".assets[] | select(.name == \"$asset_name\") | .browser_download_url" \
        2>/dev/null) || {

        # if jq isn't installed, fix that
        if ! command -v jq >/dev/null 2>&1; then
            log_error "Missing 'jq'. Install it with: sudo apt install jq"
            exit 1
        fi

        # Nice error if GitHub API complains.
        error_msg=$(printf "%s" "$response" \
            | jq -r ".message // empty" 2>/dev/null)

        if [ -n "$error_msg" ]; then
            log_error "GitHub API error: $error_msg"
        else
            log_error "Asset \"$asset_name\" not found in release v${version}."
        fi
        exit 1
    }

    log_ok "Release asset located."

    # --- temp dir -----------------------------------------------------------

    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT  # cleanup on exit

    # --- download + extract -------------------------------------------------

    log_info "Downloading binary…"
    curl -fsSL "$asset_url" -o "$tmpdir/$asset_name"
    log_ok "Downloaded."

    log_info "Extracting…"
    tar -xzf "$tmpdir/$asset_name" -C "$tmpdir"
    log_ok "Extracted."

    # --- install ------------------------------------------------------------

    log_info "Installing to ${bin_dir}…"
    install -m 755 "$tmpdir/encoderfile" "$bin_dir/encoderfile"
    log_ok "Installed successfully 🎉"

    # Warn if install directory isn't on PATH.
    if ! printf "%s" "$PATH" | tr ':' '\n' | grep -qx "$bin_dir"; then
        log_warn "${bin_dir} is not in your PATH."
        printf "\nAdd this to your shell config:\n\n"
        printf "    export PATH=\"%s:\$PATH\"\n\n" "$bin_dir"
    fi
}

# Pretty banner.
printf "%s%s%s\n" "$GREEN" "--- Encoderfile Installer | Made with <3 by Mozilla.ai ---" "$RESET"
echo ""

VERSION=latest
BIN_DIR=$(choose_install_dir)

# Argument parsing.
while [ $# -gt 0 ]; do
    case "$1" in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --bin-dir)
            BIN_DIR=$2
            shift 2
            ;;
        -*)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

cargo --version >/dev/null 2>&1 || {
    log_warn "Did not detect cargo. Please install Rust."
}

# Run installation.
install_encoderfile $VERSION $BIN_DIR
