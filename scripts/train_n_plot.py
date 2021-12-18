import os
import json
import time


# params of trainig
params = {
	"arch"       : "xc-4-4",
	"rand-shift" : 8,
	"epochs"     : 10,
	"batch-size" : 10,  
	"lr"         : 4.5e-2,  
	"wd"         : 3.0e-5,   
	"momentum"   : 0.9,   
	"eval-step"  : 10,
}
save_path = f"./tests/{params['arch']}_{params['epochs']}_SGD_lr{params['lr']}_wd{params['wd']}"
params["history"] = save_path + ".json"
params["save"] = save_path + "_model.pth" 

# training
os.system(f"python3 scripts/train.py  cifair10 \
--architecture {params['arch']} \
--rand-shift {params['rand-shift']} \
--epochs {params['epochs']} \
--batch-size {params['batch-size']} \
--lr {params['lr']} \
--weight-decay {params['wd']} \
--param momentum {params['momentum']} \
--eval-interval {params['eval-step']} \
--history {params['history']} \
--save {params['save']}")

# plot history
time.sleep(1)
os.system(f"python3 scripts/plot_training_history.py {params['history']}")

# auto-log
with open("./tests/auto_log.txt", "a") as fp:
	fp.write(f"\n\n########## {params['history'][8:-5]}")
	fp.write("\n" + json.dumps(params))
	fp.close()