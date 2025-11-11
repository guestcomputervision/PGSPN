#!/bin/bash
# Claude Code 설치 스크립트 - /login 지원
# 터미널에서 "claude code" 명령어와 /login 사용 가능

set -e

echo "=================================================="
echo "  Claude Code 설치 (/login 지원)"
echo "=================================================="
echo ""

# 1. Node.js 설치 확인
echo "1. Node.js 환경 확인..."
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js가 설치되어 있지 않습니다. 설치 중..."
    
    if command -v curl &> /dev/null; then
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    else
        echo "❌ curl이 필요합니다: sudo apt-get install curl"
        exit 1
    fi
else
    NODE_VERSION=$(node --version)
    echo "✓ Node.js 설치됨: $NODE_VERSION"
fi

NPM_VERSION=$(npm --version)
echo "✓ npm 설치됨: v$NPM_VERSION"

# 2. 공식 Claude Code 설치
echo ""
echo "2. 공식 Claude Code 설치 중..."
npm install -g @anthropic-ai/claude-code

echo "✓ Claude Code 설치 완료"

# 3. claude wrapper 설치 (claude code 명령어 지원)
echo ""
echo "3. claude wrapper 설치 중..."

cat > /tmp/claude_wrapper << 'WRAPPER_SCRIPT'
#!/bin/bash
# Claude Code Wrapper - claude code 명령어 지원

OFFICIAL_CLAUDE="/usr/bin/claude"

if [ ! -f "$OFFICIAL_CLAUDE" ]; then
    echo "❌ Claude Code가 설치되지 않았습니다."
    exit 1
fi

if [ $# -eq 0 ]; then
    # 인자 없음: 대화형 모드
    exec "$OFFICIAL_CLAUDE"
elif [ "$1" = "code" ]; then
    # claude code "프롬프트"
    if [ $# -lt 2 ]; then
        echo "사용법: claude code <프롬프트>"
        echo "예제: claude code 'Python 함수 만들어줘'"
        exit 1
    fi
    shift
    PROMPT="$*"
    echo "$PROMPT" | "$OFFICIAL_CLAUDE" --print
else
    # 다른 경우: 공식 claude에 모든 인자 전달
    exec "$OFFICIAL_CLAUDE" "$@"
fi
WRAPPER_SCRIPT

chmod +x /tmp/claude_wrapper

if [ -w /usr/local/bin ]; then
    cp /tmp/claude_wrapper /usr/local/bin/claude
    echo "✓ claude wrapper 설치 완료"
else
    sudo cp /tmp/claude_wrapper /usr/local/bin/claude
    echo "✓ claude wrapper 설치 완료"
fi

rm -f /tmp/claude_wrapper

# 4. PATH 새로고침
hash -r 2>/dev/null || true

echo ""
echo "=================================================="
echo "  ✅ 설치 완료!"
echo "=================================================="
echo ""
echo "🎯 사용법:"
echo ""
echo "  1. 로그인 (처음 사용 시):"
echo "     claude"
echo "     그 다음 터미널에서: /login"
echo ""
echo "  2. 코드 생성:"
echo "     claude code 'Python 함수 만들어줘'"
echo "     claude code 'FastAPI CRUD API'"
echo ""
echo "  3. 대화형 모드:"
echo "     claude"
echo ""
echo "=================================================="
echo ""
echo "💡 처음 사용:"
echo "   1. claude 실행"
echo "   2. /login 입력"
echo "   3. 브라우저에서 로그인"
echo "   4. 코드 작성 시작!"
echo ""
echo "=================================================="
