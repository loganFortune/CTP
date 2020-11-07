setup:
	pip3 -r requirements.txt
run:
	python3 main.py
gen-doc:
	pdoc --html main.py wvideo.py database_vision.py ./test/video_test.py --force --output-dir docs
lint-check:
	 pylint ./test/video_test.py main.py database_vision.py wvideo.py
project-test:
	python3 -m unittest test/video_test.py