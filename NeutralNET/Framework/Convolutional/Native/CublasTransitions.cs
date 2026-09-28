using NeutralNET.GPU;

namespace NeutralNET.Framework.Convolutional.Native;

public record struct CublasTransitions(CublasOperation A, CublasOperation B);
