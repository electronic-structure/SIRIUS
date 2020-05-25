# Common considerations

For the sake of consistency all parameters to C-binded functions are passed by pointer.

Inside C-api all functions are wrapped in `call_sirius` functio:

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
