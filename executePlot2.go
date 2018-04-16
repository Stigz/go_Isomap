package main

import (
    "log"
    "os"
    "os/exec"
)

func main() {
    cmd := exec.Command("python","2D_projection.py")
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    log.Println(cmd.Run())
}
