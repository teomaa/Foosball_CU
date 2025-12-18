# issac lab dir
WORKING_DIR=$PWD
ISAAC_LAB_DIR="${WORKING_DIR}/../../IsaacLab"
ROBOTS_DIR="$ISAAC_LAB_DIR/source/isaaclab_assets/isaaclab_assets/robots"
TASKS_DIR="$ISAAC_LAB_DIR/source/isaaclab_tasks/isaaclab_tasks/direct"

# check exist
if [ ! -d "$ISAAC_LAB_DIR" ]; then
  echo "❌ Isaac Lab root expected at $ISAAC_SIM_DIR"
  exit 1
else
  echo "✅ Isaac Lab found"
fi


# Ensure that the usd path (line 19) in foosball.py point to where you save the usd file foosball_no_ball.usd
USD_PATH="${PWD}/foosball_no_ball.usd"
FOOSBALL_PY="${PWD}/foosball.py"
LINE=19
NEW_LINE="        usd_path=r\"${USD_PATH}\","

sed -i "${LINE}c\\${NEW_LINE}" $FOOSBALL_PY

# copy file titled foosball.py to the IsaacLab/source/isaaclab_assets/isaaclab_assets/robots
cp FOOSBALL_PY "${ROBOTS_DIR}"
echo "✅ added foosball.py to ${ROBOTS_DIR}"

# add the line "  from .foosball import *   " on IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/__init__.py
INIT_PY="${ROBOTS_DIR}/__init__.py"
IMPORT_LINE="from .foosball import *"

grep -Fxq "$IMPORT_LINE" "$INIT_PY" || printf "\n%s\n" "$IMPORT_LINE" >> "$INIT_PY"
echo "✅ added import line to ${INIT_PY}"

# 5)Add folder titled Foosball2 to IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct
cp -r "${PWD}/foosball2" $TASKS_DIR
echo "✅ copied foosball2 to ${TASKS_DIR}"