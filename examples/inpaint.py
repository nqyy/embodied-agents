
from gradio_client import Client

client = Client("http://3.236.52.5:7860/")

result, img = client.predict(
		"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str (filepath on your computer (or URL) of image) in 'Upload' Image component
		"white wall",	# str  in 'Detection Prompt[To detect multiple objects, seperating each with '.', like this: cat . dog . chair ]' Textbox component
		"inpainting",	# str  in 'Task type' Radio component
		"windows",	# str  in 'Inpaint/Outpaint Prompt (if this is empty, then remove)' Textbox component
		0.3,	# int | float (numeric value between 0.0 and 1.0) in 'Box Threshold' Slider component
		0.25,	# int | float (numeric value between 0.0 and 1.0) in 'Text Threshold' Slider component
		0.8,	# int | float (numeric value between 0.0 and 1.0) in 'IOU Threshold' Slider component
		"merge",	# str  in 'inpaint_mode' Radio component
		"type what to detect below",	# str  in 'Mask from' Radio component
		"segment",	# str  in 'remove mode' Radio component
		"10",	# str  in 'remove_mask_extend' Textbox component
		1,	# int | float (numeric value between 1 and 20) in 'How many relations do you want to see' Slider component
		"Brief",	# str  in 'Kosmos Description Type' Radio component
		fn_index=2
)
print(type(result))
print(img)
