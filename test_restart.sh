#!/bin/bash

# Start supervisord in background
supervisord -c supervisord.conf &
SUPERVISOR_PID=$!

# Wait for it to start
sleep 5

# Check if streamlit is running
STREAMLIT_PID=$(pgrep -f "streamlit run app.py")

if [ -z "$STREAMLIT_PID" ]; then
  echo "Streamlit not running initially"
  kill $SUPERVISOR_PID
  exit 1
fi

echo "Initial Streamlit PID: $STREAMLIT_PID"

# Kill streamlit to simulate crash
kill $STREAMLIT_PID

# Wait for restart
sleep 5

# Check if restarted
NEW_STREAMLIT_PID=$(pgrep -f "streamlit run app.py")

if [ -z "$NEW_STREAMLIT_PID" ]; then
  echo "Streamlit not restarted"
else
  echo "Streamlit restarted with PID: $NEW_STREAMLIT_PID"
fi

# Kill supervisord
kill $SUPERVISOR_PID

# Show logs
echo "Streamlit stdout log:"
cat logs/streamlit.log

echo "Streamlit stderr log:"
cat logs/streamlit.err