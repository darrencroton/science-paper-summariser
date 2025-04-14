#!/bin/zsh

cd "${0:a:h}"  # Change to script directory using zsh syntax

# Stop paper summarisation from PID file
if [ -f logs/process.pid ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Stopping paper summarisation process $(cat logs/process.pid)" >> logs/history.log
    kill $(cat logs/process.pid) 2>/dev/null
    rm logs/process.pid
fi

# Also check for any other running summarise.py processes
OTHER_PIDS=$(ps aux | grep "[p]ython3 summarise.py" | awk '{print $2}')
if [ ! -z "$OTHER_PIDS" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Found additional summarisation processes: $OTHER_PIDS" >> logs/history.log
    for pid in $OTHER_PIDS; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Stopping additional process $pid" >> logs/history.log
        kill $pid 2>/dev/null
    done
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - All paper summarisation processes stopped\n" >> logs/history.log