```C++

//首先分成2048个组，每组包含多个数值（基本都是小数）
Input: FP32 histogram H with 2048 bins: bin[ 0 ], …, bin[ 2047 ]

For i in range( 128 , 2048 ): // |T|的取值肯定在第128-2047组之间,取每组的中点
    reference_distribution_P = [ bin[ 0 ] , ..., bin[ i-1 ] ] // 选取前 i 组构成P，i>=128
    outliers_count = sum( bin[ i ] , bin[ i+1 ] , … , bin[ 2047 ] ) //边界外的组
    reference_distribution_P[ i-1 ] += outliers_count //边界外的组加到边界P[i-1]上，没有直接丢掉（概率为1）
    P /= sum(P) // 归一化
    
    // 将前面的P（包含i个组，i>=128），映射到 0-128 上，映射后的称为Q，Q包含128个组，
    // 一个整数是一组
    candidate_distribution_Q = quantize [ bin[ 0 ], …, bin[ i-1 ] ] into 128 levels
    
    //这时的P（包含i个组，i>=128）和Q向量（包含128个组）的大小是不一样的，无法直接计算二者的KL散度
    //因此需要将Q扩展为 i 个组，以保证跟P大小一样
    expand candidate_distribution_Q to ‘ i ’ bins
    
    Q /= sum(Q) // 归一化
    //计算P和Q的KL散度
    divergence[ i ] = KL_divergence( reference_distribution_P, candidate_distribution_Q)
End For

//找出 divergence[ i ] 最小的数值，假设 divergence[m] 最小，
//那么|T|=( m + 0.5 ) * ( width of a bin )
Find index ‘m’ for which divergence[ m ] is minimal
threshold = ( m + 0.5 ) * ( width of a bin )

```
