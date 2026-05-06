#!/bin/zsh

cd "${0:a:h}"

exec ./start_paper_summariser.sh cli copilot claude-sonnet-4.6 --effort high
