__kernel void demo(
    int a
)
{
    printf("%d\n", get_global_id(0));
}