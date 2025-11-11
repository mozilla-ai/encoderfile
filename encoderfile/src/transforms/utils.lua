function CreateGoodTable()
    return {
        {1, 2, 3},
        {4, 5.0, 6},
        {7, 8, 9}
    }
end

function CreateRaggedTable()
    return {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8}
    }
end

function CreateStringTable()
    return { "hello", "i", "am", "not", "numbers" }
end

function TestEq(arr1, arr2)
    return arr1 == arr2
end

function TestAddition(arr1, arr2)
    return arr1 + arr2
end

function TestSubtraction(arr1, arr2)
    return arr1 - arr2
end

function TestMultiplication(arr1, arr2)
    return arr1 * arr2
end

function TestDivision(arr1, arr2)
    return arr1 / arr2
end
