import os

# Remove quotations in path
path_to_log = input("Enter path to log file: ").removeprefix('"').removesuffix('"')

POSITION_LOGS_UUID = "72b8f0c1-bdb8-42d2-81c6-ca32bbb0a6b0"

# Open the file in read mode
with open(path_to_log, "r") as input:

    os.makedirs("target", exist_ok=True)  # make directory if it does not exist

    output_log = open(f"target/{os.path.basename(path_to_log)}", "w+")

    timestamp = 0

    # Loop through each line in the file
    for line in input:
        # Process the line here
        stripped = line.strip()  # remove any leading or trailing whitespace
        if stripped.startswith('"sandboxLog"'):
            output_log.write(
                f"=================   TIMESTAMP {timestamp} ================\n\n"
            )
            timestamp += 100
            output_log.write("Sandbox Log:\n")
        elif stripped.startswith('"lambdaLog"'):
            output_log.write("Lambda Log:\n")
        elif stripped == "":
            break
        else:
            continue

        index = stripped.find(":")
        log = stripped[index + 1 :].lstrip()

        # get rid of leading and trailing \"
        if log[-1] == ",":
            log = log[:-1]
        log = log[1:]
        log = log[:-1]

        log = log.replace("\\n", "\n")
        log = log.replace("\\t", "\t")
        output_log.write(log + "\n\n")

    output_log.close()
input.close()


with open(f"target/{os.path.basename(path_to_log)}", "r") as input:
    os.makedirs("target/csv", exist_ok=True)  # make directory if it does not exist

    output_position_csv = open(f"target/csv/{os.path.basename(path_to_log)}", "w+")

    timestamp = 0

    output_position_csv.write("uuid,timestamp,position,product")
    # Loop through each line in the file
    for line in input:
        if POSITION_LOGS_UUID in line:
            output_position_csv.write(line)
        # Process the line here

    output_position_csv.close()
input.close()
