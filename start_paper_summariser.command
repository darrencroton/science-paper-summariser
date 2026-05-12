#!/bin/zsh

cd "${0:a:h}"

if [[ -r "$HOME/.llm/.env.llm" ]]; then
  source "$HOME/.llm/.env.llm"
fi

export OPENAI_COMPATIBLE_BASE_URL="https://djcmacstudio.tail98bbb1.ts.net/v1"
export OPENAI_COMPATIBLE_API_KEY_ENV="LOCAL_LLM_API_KEY"

exec ./start_paper_summariser.sh api openai-compatible minimax/minimax-m2.7-q8
