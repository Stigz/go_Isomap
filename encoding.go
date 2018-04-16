package main

import (
    "os"
    "log"
    "encoding/csv"
    "strconv"
)

func EncodeCSV3D(X, Y, Z []float64) {
    data := make([][]string, 2)
    data[0] = make([]string, 0)
    data[1] = make([]string, 0)
    data[2] = make([]string, 0)

    for i := 0; i < len(X); i++ {
        data[0] = append(data[0], strconv.FormatFloat(X[i], 'f', -1, 64))
        data[1] = append(data[1], strconv.FormatFloat(Y[i], 'f', -1, 64))
        data[2] = append(data[2], strconv.FormatFloat(Z[i], 'f', -1, 64))
    }

    file, err := os.Create("result.csv")
    checkError("Cannot create file", err)
    defer file.Close()

    writer := csv.NewWriter(file)
    defer writer.Flush()

    for _, value := range data {
        err := writer.Write(value)
        checkError("Cannot write to file", err)
    }
}

func EncodeCSV2D(X, Y []float64) {
    data := make([][]string, 2)
    data[0] = make([]string, 0)
    data[1] = make([]string, 0)

    for i := 0; i < len(X); i++ {
        data[0] = append(data[0], strconv.FormatFloat(X[i], 'f', -1, 64))
        data[1] = append(data[1], strconv.FormatFloat(Y[i], 'f', -1, 64))
    }

    file, err := os.Create("result2.csv")
    checkError("Cannot create file", err)
    defer file.Close()

    writer := csv.NewWriter(file)
    defer writer.Flush()

    for _, value := range data {
        err := writer.Write(value)
        checkError("Cannot write to file", err)
    }
}

func main(){
    
}

func checkError(message string, err error) {
    if err != nil {
        log.Fatal(message, err)
    }
}