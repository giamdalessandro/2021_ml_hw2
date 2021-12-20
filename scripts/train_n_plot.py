import os
import json
import time

PLOT = True

# params of trainig
params = {
	"arch"       : "wrn-16-10",
	"rand-shift" : 4,
	"epochs"     : 200,
	"batch-size" : 10,  
	"lr"         : 4.55e-3,  
	"wd"         : 5.29e-3,   
	"momentum"   : 0.9,   
	"eval-step"  : 20,
}
save_path = f"./tests/{params['arch']}_{params['epochs']}_SGD_lr{params['lr']}_wd{params['wd']}_rand{params['rand-shift']}"
params["history"] = save_path + ".json"
params["save"] = save_path + "_model.pth" 

# training
print(f"testing model {params['history'][8:-5]} ...")
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
if PLOT: os.system(f"python3 scripts/plot_training_history.py {params['history']}")

# auto-log
with open(f"./logs/{params['arch'][:3]}-auto_log.txt", "a") as fp:
	fp.write(f"\n\n########## {params['history'][8:-5]}")
	fp.write("\n" + json.dumps(params))
	fp.close()