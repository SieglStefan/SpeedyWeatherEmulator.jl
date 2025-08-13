"""
    add_two(x::T) where {T<:Number} 
Adds two to the input number .
"""
 function add_two(x::T) where {T<:Number}
    x + T(2) 
end