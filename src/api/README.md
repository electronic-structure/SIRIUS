# Common considerations

By default, all variables are passed by pointer. There are two exceptions to this rule: 
  - all input, required, simple data types (bool, int, double) are passed by value.
  - function pointers are passed by value

Inside C-api all functions are wrapped in `call_sirius` function:

```
call_sirius([&]()
{
  // actual call to sirius 
}, error_code__);
```

On the Fortran side all pointers are declared as
```
type(C_PTR) :: var
```
