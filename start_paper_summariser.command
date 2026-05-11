#!/bin/zsh

cd "${0:a:h}"

OPENAI_COMPATIBLE_BASE_URL=http://localhost:1234/v1 exec ./start_paper_summariser.sh api openai-compatible minimax/minimax-m2.7
