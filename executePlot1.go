package main

import (
    "log"
    "os"
    "os/exec"
)

func main() {
    cmd := exec.Command("python","lines3d_demo.py")
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    log.Println(cmd.Run())
}
