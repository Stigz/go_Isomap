package main

import (
    "fmt"
    "math/rand"
    "math/cmplx"
    "math"
    "gonum.org/v1/gonum/mat"
    //"gonum.org/v1/gonum/blas"
    "strconv"
    "os"
    "log"
    "encoding/csv"
    "sort"
    "time"
)

type Slice struct {
    sort.Float64Slice
    idx []int
}

func (s Slice) Swap(i, j int) {
    s.Float64Slice.Swap(i, j)
    s.idx[i], s.idx[j] = s.idx[j], s.idx[i]
}

func NewSlice(n []float64) *Slice {
    s := &Slice{Float64Slice: sort.Float64Slice(n), idx: make([]int, len(n))}
    for i := range s.idx {
        s.idx[i] = i
    }
    return s
}

func EncodeCSV3D(X, Y, Z, V []float64) {
    data := make([][]string, 4)
    data[0] = make([]string, 0)
    data[1] = make([]string, 0)
    data[2] = make([]string, 0)
    data[3] = make([]string, 0)

    for i := 0; i < len(X); i++ {
        data[0] = append(data[0], strconv.FormatFloat(X[i], 'f', -1, 64))
        data[1] = append(data[1], strconv.FormatFloat(Y[i], 'f', -1, 64))
        data[2] = append(data[2], strconv.FormatFloat(Z[i], 'f', -1, 64))
        data[3] = append(data[2], strconv.FormatFloat(V[i], 'f', -1, 64))
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
    data := make([][]string, 4)
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

func checkError(message string, err error) {
    if err != nil {
        log.Fatal(message, err)
    }
}

//Square two matrices elementwise (each element is square of itself)
func squareElem(array []float64, n int) []float64 {
    result := make([]float64, 0)
    for i:=0; i<n; i++ {
        result = append(result, array[i]*array[i])
    }
    return result
}

func squareRoot(i, j int, v float64) float64 {
    return float64(math.Sqrt(v))
}

func power(i, j int, v float64) float64 {
    return math.Pow(v,2.0)
}

//Compute the pairwise euclidean distance between points
func euclDist(X, Y, Z []float64, n int) mat.Matrix {
    //Square elements
    X1 := squareElem(X, n)
    Y1 := squareElem(Y, n)
    Z1 := squareElem(Z, n)

    D1 := make([]float64, 0)
    for i:=0; i<n; i++ {
        D1 = append(D1, X1[i]+Y1[i]+Z1[i])
    }

    temp_D1 := D1
    for i:=0; i<(n-1); i++ {
        D1 = append(D1,temp_D1...)
    }


    D1_Mat := mat.NewDense(n,n,D1)
    D1_Mat.Add(D1_Mat,D1_Mat.T())

    D2 := make([]float64, 0)
    D2 = append(D2, X...)
    D2 = append(D2, Y...)
    D2 = append(D2, Z...)

    D3 := make([]float64, 0)
    D3 = append(D3, X...)
    D3 = append(D3, Y...)
    D3 = append(D3, Z...)
    
    D2_Mat := mat.NewDense(3,n,D2)
    D2_Scaled := mat.NewDense(3,n,D3)
    D2_Scaled.Scale(2, D2_Scaled)

    D := mat.NewDense(n, n, nil)
    D.Product(D2_Mat.T(),D2_Scaled)

    D.Sub(D1_Mat, D)
    D.Apply(squareRoot, D)

    return D
}

func knn(D_eucl mat.Matrix, n, k int) *mat.Dense {

    D_slice := make([][]float64, n)
    for i := 0; i<n; i++ {
        temp := make([]float64, n)
        D_slice[i] = mat.Col(temp, i, D_eucl)
    }

    //sort columns by increasing value
    knnVals := make([]float64, 0)
    knnIdxs := make([]int, 0)
    for i := range D_slice {
        s := NewSlice(D_slice[i])
        sort.Sort(s)
        s.Float64Slice = s.Float64Slice[1:(k+1)]
        s.idx = s.idx[1:(k+1)]
        knnVals = append(knnVals, s.Float64Slice...)
        knnIdxs = append(knnIdxs, s.idx...)
    }

    B := make([]int, 0)
    for i := 0; i < n; i++ {
        for j :=0; j < k; j++{
            B = append(B, i)
        }
    }

    //Populate a matrix using the B array as the row index, the Idxs array as 
    //the column index and the knnVals as the value at that index in the matrix
    var counter = 0
    D_floyd := mat.NewDense(n,n,nil)
    for i:=0; i<(n*k); i++ {
        D_floyd.Set(B[counter],knnIdxs[counter],knnVals[counter])
        counter = counter +1
    }
    return D_floyd
}

func floyd(D_floyd *mat.Dense, n int) *mat.Dense {
    D_floyd.Add(D_floyd, D_floyd.T())
    D_floyd.Scale(.5, D_floyd)

    //TODO: try changing large distance to infinity
    //keep diagonals at 0 distance (nodes connected to themselves), 
    //nodes not connected have (almost) infinity distance
    for i:=0; i<n;i++{
        for j:=0; j<n;j++{
            if i==j {
                continue
            } else if D_floyd.At(i,j) == 0.0 {
                D_floyd.Set(i,j,10000)
            }
        }
    }

    //Floyd-Warshall Algorithm
    for k:=0; k<n;k++{
        for i:=0; i<n;i++{
            for j:=0; j<n;j++{
                if D_floyd.At(i,j) > (D_floyd.At(i,k) + D_floyd.At(k,j)){
                    D_floyd.Set(i,j,(D_floyd.At(i,k) + D_floyd.At(k,j)))
                }
            }
        }
    }

    //For each diagonal (connection to itself - which is actually 0)
    //we need to add values for distance traveled  from node i to closest
    //node j and back to node i again so that next part works
    mins := make([]float64,0)
    for i := 0; i<n; i++ {
        temp := make([]float64, n)
        temp = mat.Col(temp, i, D_floyd)
        var min = 1000000.0
        for j := 0; j<n; j++ {
            if temp[j] < min && temp[j] != 0.0 {
                min = temp[j]
            }
        }
        mins = append(mins, min)
    }

    for i:=0; i<n;i++{
        D_floyd.Set(i,i,2*mins[i])
    }
    return D_floyd
}

func mds(D_mds *mat.Dense, n int) (firstVect, secondVect []float64) {
    ones := mat.NewDense(n,n,nil)
    for i:=0; i<n; i++{
        ones.Set(i,i,1)
    }

    J_s := make([]float64, 0)
    for i := 0; i < n*n; i++{
        J_s = append(J_s, 1.0/float64(n))
    }
    J := mat.NewDense(n, n, J_s)

    J.Sub(ones,J)

    D_mds.Apply(power,D_mds)
    D_mds.Mul(J,D_mds)
    D_mds.Mul(D_mds,J)
    K := mat.NewDense(n, n, nil)
    K.Scale(-0.5,D_mds)

    e := &mat.Eigen{}
    e.Factorize(K,false,true)
    valsTemp := make([]complex128, n)
    eigenVals := e.Values(valsTemp)
    //vectsTemp := mat.NewDense(n,n,nil)
    eigenVects := e.Vectors()

        
    daFloats := make([]float64,0)
    for i:=0; i<n;i++{
        temp := eigenVals[i]
        neg := cmplx.Phase(temp) //test to see what phase its in to negate answer
        if neg == 0.0 {
            daFloats = append(daFloats, cmplx.Abs(temp))
        } else {
            daFloats = append(daFloats, -cmplx.Abs(temp)) //If
        }
    }

    max := 0.0
    max2 := 0.0
    var firstIndex = -1
    var secondIndex = -1

    for i:=0; i<n;i++{
        if max < daFloats[i] {
            max = daFloats[i]
            daFloats[i] = 0
            firstIndex = i
        }
        if max2 < daFloats[i] {
            max2 = daFloats[i]
            secondIndex = i
        }
    }

    temp := make([]float64,n)
    firstVect = mat.Col(temp, firstIndex, eigenVects)
    temp = make([]float64, n)
    secondVect = mat.Col(temp, secondIndex, eigenVects)
    
    //possibly negate (same eigenvector)
    for i:=0; i<n;i++{
        firstVect[i] = math.Pow(max,.5)*firstVect[i] 
        secondVect[i] = math.Pow(max2,.5)*secondVect[i]
    }
    return firstVect, secondVect
}

func main() {
    start := time.Now()

    // Create Swiss Roll Data Set
    //Number of points
    var n = 1000

    
    //Create a random 2xn matrix of floats between 0 and 1
    data := make([]float64, 2*n)

    rand.Seed(68)
    for i := range data {
        data[i] = rand.Float64()
    }
    

    //data := []float64{0.9746, 0.7158, 0.8360, 0.7816, 0.3263, 0.1683, 0.5324, 0.0718, 0.6030, 0.7271, 0.1061, 0.9565, 0.3980, 0.4715, 0.2384, 0.1405, 0.0071, 0.8520, 0.3958, 0.0321}
    a := mat.NewDense(2, n, data)

    //Isolate the first row of the random numbers
    f := make([]float64, n)
    firstRow := mat.Row(f, 0, a)

    //Apply some crazy math to generate the Swiss roll eventually
    for i := range firstRow {
        firstRow[i] = (2*firstRow[i] + .1)*3*math.Pi/2
    }

    //Setup the X, Y and Z values for the Swiss roll coordinates
    //The X and Z coordinates are collinear while the Y is independent/random
    //Hence, we want to find 2D representation of 3D Swiss roll
    X := make([]float64, 0)
    y_temp := make([]float64, n)
    Y := mat.Row(y_temp, 1, a) //The y axis is the second row of random numbers
    Z := make([]float64, 0)


    for i := range firstRow {
        var tempX = -math.Cos(firstRow[i])*firstRow[i]
        X = append(X, tempX)
    }

    for i := range Y {
        Y[i] = 20*Y[i]
    }

    for i := range firstRow {
        var tempZ = math.Sin(firstRow[i])*firstRow[i]
        Z = append(Z, tempZ)
    }
    
    D_eucl := euclDist(X,Y,Z,n)


    //SORTING k-nn return matrix for Floyd-Warshall Algorithm manipulation
    var k = 6
    D_floyd := knn(D_eucl, n, k)

    //Floyd-Warshall algorithm returns matrix ready for Multimensional Scaling
    D_mds := floyd(D_floyd, n)
    
    //MDS
    firstVect, secondVect := mds(D_mds, n)

   // fmt.Println(firstVect,secondVect)
    fmt.Println(firstVect)
    EncodeCSV2D(firstVect,secondVect)
    EncodeCSV3D(X,Y,Z,firstRow)
    //Creating the 3xn matrix of values generated for the Swiss roll
   // SwissRoll := mat.NewDense(3, n, nil)
    elapsed := time.Since(start)
    log.Printf("Main took %s", elapsed)
}
