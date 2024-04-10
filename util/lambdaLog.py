import os

path_to_log = input("Enter path to log file: ")

# Open the file in read mode
with open(path_to_log, "r") as input:

    os.makedirs("target", exist_ok=True)  # make directory if it does not exist

    with open(f"target/{os.path.basename(path_to_log)}", "w+") as output:

        timestamp = 0

        # Loop through each line in the file
        for line in input:
            # Process the line here
            stripped = line.strip()  # remove any leading or trailing whitespace
            if stripped == "":
                break
            elif stripped.startswith('"lambdaLog"'):
                output.write(f"########         TIMESTAMP {timestamp}       ########\n")
                index = stripped.find(":")
                log = stripped[index + 1 :].lstrip()

                # get rid of leading and trailing \"
                log = log[1:]
                log = log[:-1]

                log = log.replace("\\n", "\n")
                log = log.replace("\\t", "\t")
                output.write(log + "\n\n")
                timestamp += 100

    output.close()
input.close()
