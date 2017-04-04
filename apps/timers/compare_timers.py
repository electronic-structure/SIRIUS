import json
import sys

def main():

    if (len(sys.argv) < 3):
        print("Usage: %s file1.json file2.json"%sys.argv[0])
        sys.exit(1)

    jf1 = json.load(open(sys.argv[1], 'r'))
    jf2 = json.load(open(sys.argv[2], 'r'))
    
    for key in jf1["timers"]:
        t1 = float(jf1["timers"][key][0])
        if key in jf2["timers"]:
            t2 = float(jf2["timers"][key][0])
            if (t2 > 1):
                print("%f  %s  %f %f"%(t1 / t2, key, t1, t2))

if __name__ == "__main__":
    main()
