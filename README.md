mkdir grounding_project
cd grounding_project
code .

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

python3 -m venv gd_env
source gd_env/bin/activate


pip install --upgrade pip
pip install -r requirements.txt

mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swint_ogc.pth
cd ..
