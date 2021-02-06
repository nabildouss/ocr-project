from src import ctc_decoder

def predict(x, model, framework='torch'):
	if framework == 'torch':
		log_P = model(x)
		return ctc_decoder.torch_confidence(log_P=x)
	elif framework == 'clstm':
		pass
	else:
		ValueError(f'unknown framework: {framework}\nPlease choose between torch or clstm')
