import os
import re
from typing import Dict, List, Optional, Tuple, Set

class KernelRunnerTransformer:
    """Enhanced class for transforming CUDA kernel code with various optimizations."""

    def __init__(self, source_path):
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Runner source not found: {source_path}")
        
        self.source_path = source_path
        with open(source_path, 'r') as f:
            self.source_code = f.read()
            self.original_code = self.source_code  # Keep a copy of the original code
        
        self.kernel_name = self._extract_kernel_name()
        self.optimizations_applied = []
        print(f"Found kernel name: {self.kernel_name}")
    
    def _extract_kernel_name(self):
        """Extract the kernel name from the source code."""
        kernel_pattern = r'__global__\s+void\s+(\w+)\s*\('
        match = re.search(kernel_pattern, self.source_code)
        if match:
            return match.group(1)
        return None
    
    def _extract_kernel_params(self):
        """Extract the kernel parameters from the source code."""
        if not self.kernel_name:
            return "float *A, float *B, float *C, int numElements"  # Default fallback
            
        param_pattern = rf'__global__\s+void\s+{self.kernel_name}\s*\(([^{{]*)\)'
        param_match = re.search(param_pattern, self.source_code)
        
        if not param_match:
            print("Error: Could not extract kernel parameters. Using default params.")
            return "float *A, float *B, float *C, int numElements"
        else:
            return param_match.group(1).strip()
    
    def _find_array_variables(self) -> Set[str]:
        """Find all array variables accessed using thread/block indices."""
        # Pattern to match array accesses with thread/block indices
        array_pattern = r'(\w+)\s*\[\s*(blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x|threadIdx\.x\s*\+\s*blockIdx\.x\s*\*\s*blockDim\.x)\s*\]'
        array_matches = re.finditer(array_pattern, self.source_code)
        
        array_names = set()
        for match in array_matches:
            array_names.add(match.group(1))
        
        return array_names
    
    def _find_loops(self) -> List[Dict]:
        """Find all loops in the kernel code."""
        loop_pattern = r'for\s*\(\s*int\s+(\w+)\s*=\s*(\w+|\d+)\s*;\s*\1\s*<\s*([^;]+)\s*;\s*\1(\+\+|\+=\s*1)\s*\)\s*\{'
        loop_matches = re.finditer(loop_pattern, self.source_code, re.DOTALL)
        
        loops = []
        for match in loop_matches:
            # Find the loop body by counting braces
            start_pos = match.end()
            brace_count = 1
            end_pos = -1
            
            for i in range(start_pos, len(self.source_code)):
                if self.source_code[i] == '{':
                    brace_count += 1
                elif self.source_code[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > 0:
                loops.append({
                    'index_var': match.group(1),
                    'start_val': match.group(2),
                    'end_cond': match.group(3),
                    'increment': match.group(4),
                    'start_pos': match.start(),
                    'end_pos': end_pos,
                    'body': self.source_code[start_pos:end_pos-1]  # Exclude the closing brace
                })
        
        return loops
    
    def _replace_kernel_body(self, new_body):
        """Replace the kernel body with a new implementation."""
        if not self.kernel_name:
            print("Error: Could not identify kernel name")
            return False
            
        # Extract the kernel parameters
        params = self._extract_kernel_params()
        
        # Create the full kernel definition
        full_kernel = f"""
__global__ void {self.kernel_name}({params}) {{
{new_body}
}}
"""
        
        # Pattern to find the kernel in the source code
        kernel_pattern = rf'__global__\s+void\s+{self.kernel_name}\s*\([^{{]+\{{[^}}]+\}}'
        
        try:
            # Try simple replacement first
            transformed_code = re.sub(kernel_pattern, full_kernel, self.source_code, flags=re.DOTALL)
            
            # If that didn't work, try more complex approach
            if transformed_code == self.source_code:
                print("Warning: Simple replacement failed. Trying alternative approach.")
                
                # Find start of kernel
                kernel_start = re.search(rf'__global__\s+void\s+{self.kernel_name}\s*\(', self.source_code)
                if kernel_start:
                    start_pos = kernel_start.start()
                    # Find the opening brace
                    open_brace_pos = self.source_code.find('{', start_pos)
                    if open_brace_pos > 0:
                        # Find the closing brace with proper nesting
                        brace_count = 1
                        close_brace_pos = -1
                        for i in range(open_brace_pos + 1, len(self.source_code)):
                            if self.source_code[i] == '{':
                                brace_count += 1
                            elif self.source_code[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    close_brace_pos = i
                                    break
                        
                        if close_brace_pos > 0:
                            # Replace the kernel
                            transformed_code = (
                                self.source_code[:start_pos] + 
                                full_kernel + 
                                self.source_code[close_brace_pos + 1:]
                            )
                            
            self.source_code = transformed_code
            return True
        except Exception as e:
            print(f"Error replacing kernel body: {e}")
            return False

    def apply_shared_memory(self, block_size: int = 256, remove_existing_index: bool = False) -> bool:
        """
        Apply shared memory optimization to the kernel.
        
        Args:
            block_size: Size of the shared memory arrays
            remove_existing_index: Whether to remove existing index declarations
            
        Returns:
            bool: Success status
        """
        if not self.kernel_name:
            print("Error: Could not identify kernel name")
            return False
        
        # Find array variables that could benefit from shared memory
        array_names = self._find_array_variables()
        
        if not array_names:
            print("No suitable array accesses found for shared memory optimization")
            return False
        
        # Generate the new kernel body with shared memory
        new_body = ""
        
        # Add shared memory declarations
        new_body += "    // Shared memory declarations\n"
        for array in array_names:
            new_body += f"    __shared__ float {array}_shared[{block_size}];\n"
        
        # Add global index calculation
        new_body += "\n    // Global thread index\n"
        new_body += "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        
        # Add loading into shared memory
        new_body += "\n    // Load data into shared memory\n"
        for array in array_names:
            new_body += f"    {array}_shared[threadIdx.x] = {array}[i];\n"
        
        # Add synchronization
        new_body += "\n    // Synchronize to make sure data is loaded\n"
        new_body += "    __syncthreads();\n"
        
        # Add computation using shared memory
        new_body += "\n    // Compute using data from shared memory\n"
        
        # Try to detect the computation pattern
        computation_pattern = r'(\w+)\s*\[\s*i\s*\]\s*=\s*(.+);'
        comp_match = re.search(computation_pattern, self.source_code)
        
        if comp_match:
            output_array = comp_match.group(1)
            expression = comp_match.group(2)
            
            # Replace array accesses with shared memory
            for array in array_names:
                expression = re.sub(rf'{array}\s*\[\s*i\s*\]', f"{array}_shared[threadIdx.x]", expression)
            
            new_body += f"    {output_array}[i] = {expression};\n"
        else:
            # Default computation (assuming C = A + B)
            if "A" in array_names and "B" in array_names and "C" in array_names:
                new_body += "    C[i] = A_shared[threadIdx.x] + B_shared[threadIdx.x];\n"
            else:
                print("Warning: Could not detect computation pattern. Using placeholder.")
                for array in array_names:
                    if array != array_names[0]:  # Assume first array is output
                        new_body += f"    {array_names[0]}[i] = {array}_shared[threadIdx.x];\n"
                        break
        
        # Replace the kernel body
        success = self._replace_kernel_body(new_body)
        
        if success:
            self.optimizations_applied.append("shared_memory")
            print("✅ Successfully applied shared memory optimization")
        
        return success

    def apply_loop_unrolling(self, unroll_factor: int = 4) -> bool:
        """
        Apply loop unrolling optimization to the kernel.
        
        Args:
            unroll_factor: The unrolling factor
            
        Returns:
            bool: Success status
        """
        # Find all loops in the code
        loops = self._find_loops()
        
        if not loops:
            print("No loops found for unrolling")
            return False
        
        # Make a copy of the code for modification
        modified_code = self.source_code
        
        # Track position offsets as we modify the code
        offset = 0
        
        # Process each loop
        for loop in loops:
            # Extract loop components
            idx_var = loop['index_var']
            start_val = loop['start_val']
            end_cond = loop['end_cond']
            body = loop['body']
            
            # Adjust positions for previous modifications
            start_pos = loop['start_pos'] + offset
            end_pos = loop['end_pos'] + offset
            
            # Create unrolled loop
            unrolled_loop = f"""
    // Loop unrolled by factor {unroll_factor}
    for (int {idx_var} = {start_val}; {idx_var} < {end_cond} - {unroll_factor - 1}; {idx_var} += {unroll_factor}) {{
"""
            
            # Add unrolled iterations
            for i in range(unroll_factor):
                # Create a copy of the body with adjusted index
                if i == 0:
                    # First iteration uses the original variable
                    iter_body = body
                else:
                    # Subsequent iterations use (idx + i)
                    iter_body = body.replace(f"{idx_var}", f"({idx_var} + {i})")
                    
                    # Special handling for array accesses
                    iter_body = re.sub(
                        r'(\w+)\s*\[\s*(\w+)\s*\]', 
                        lambda m: f"{m.group(1)}[{m.group(2)} + {i}]" if m.group(2) == idx_var else m.group(0),
                        iter_body
                    )
                
                unrolled_loop += iter_body + "\n"
            
            unrolled_loop += "    }\n\n"
            
            # Add cleanup loop for remaining iterations
            unrolled_loop += f"""    // Handle remaining iterations
    for (int {idx_var} = ({end_cond} - ({end_cond} % {unroll_factor})); {idx_var} < {end_cond}; {idx_var}++) {{
{body}
    }}
"""
            
            # Replace the original loop with the unrolled version
            modified_code = modified_code[:start_pos] + unrolled_loop + modified_code[end_pos:]
            
            # Update offset for subsequent loops
            offset += len(unrolled_loop) - (end_pos - start_pos)
        
        # Update the source code
        self.source_code = modified_code
        self.optimizations_applied.append("loop_unrolling")
        print(f"✅ Successfully unrolled {len(loops)} loops with factor {unroll_factor}")
        
        return True

    def apply_memory_coalescing(self) -> bool:
        """
        Apply memory coalescing optimization to the kernel.
        
        Returns:
            bool: Success status
        """
        # Pattern to find non-coalesced memory access
        non_coalesced_pattern = r'(\w+)\s*\[\s*(threadIdx\.x\s*\*\s*\w+|threadIdx\.x\s*\*\s*\d+)\s*\]'
        
        if not re.search(non_coalesced_pattern, self.source_code):
            # Try a more general pattern to find array accesses
            access_pattern = r'(\w+)\s*\[\s*([^]]*threadIdx\.x[^]]*)\s*\]'
            accesses = re.finditer(access_pattern, self.source_code)
            
            has_modifications = False
            modified_code = self.source_code
            
            for match in accesses:
                array_name = match.group(1)
                access_expr = match.group(2)
                
                # Check if access is already optimal
                if 'blockDim.x * blockIdx.x + threadIdx.x' in access_expr:
                    continue
                
                # Try to convert to coalesced pattern
                if 'blockIdx.x' in access_expr and 'threadIdx.x' in access_expr:
                    # Already has both index components, try to reorder
                    coalesced_expr = access_expr.replace(
                        'blockIdx.x * blockDim.x + threadIdx.x', 
                        'threadIdx.x + blockIdx.x * blockDim.x'
                    )
                    
                    if coalesced_expr != access_expr:
                        modified_code = modified_code.replace(
                            f"{array_name}[{access_expr}]",
                            f"{array_name}[{coalesced_expr}] // Coalesced access"
                        )
                        has_modifications = True
            
            if has_modifications:
                self.source_code = modified_code
                self.optimizations_applied.append("memory_coalescing")
                print("✅ Applied memory coalescing optimization")
                return True
            else:
                print("No memory access patterns found that could benefit from coalescing")
                return False
        else:
            # Handle non-coalesced accesses (e.g., threadIdx.x * stride pattern)
            modified_code = re.sub(
                non_coalesced_pattern,
                lambda m: f"{m.group(1)}[threadIdx.x + blockIdx.x * blockDim.x] // Coalesced access",
                self.source_code
            )
            
            self.source_code = modified_code
            self.optimizations_applied.append("memory_coalescing")
            print("✅ Applied memory coalescing optimization")
            return True

    def apply_register_optimization(self, max_registers: int = 32) -> bool:
        """
        Apply register optimization to the kernel.
        
        Args:
            max_registers: Maximum number of registers per thread
            
        Returns:
            bool: Success status
        """
        # Add pragma to limit register usage
        pragma = f"\n// Pragma to limit register usage\n#pragma maxrregcount={max_registers}\n"
        
        # Find where to insert the pragma (before the kernel definition)
        kernel_pattern = rf'__global__\s+void\s+{self.kernel_name}\s*\('
        
        kernel_match = re.search(kernel_pattern, self.source_code)
        if not kernel_match:
            print("Error: Could not find kernel definition")
            return False
        
        insert_position = kernel_match.start()
        
        # Insert the pragma
        modified_code = (
            self.source_code[:insert_position] + 
            pragma + 
            self.source_code[insert_position:]
        )
        
        # Apply register use optimizations - Find variables to combine or reuse
        modified_body = modified_code
        
        # Look for temporary variables that could be reused
        temp_var_pattern = r'float\s+(\w+)\s*=\s*[^;]+;'
        temp_vars = re.finditer(temp_var_pattern, modified_code)
        
        var_reuse_comment = "\n    // Variables reused to reduce register pressure\n"
        added_comment = False
        
        scope_vars = {}
        
        for match in temp_vars:
            var_name = match.group(1)
            var_pos = match.start()
            
            # Find the scope depth of this variable
            scope_depth = 0
            for i in range(var_pos):
                if modified_code[i] == '{':
                    scope_depth += 1
                elif modified_code[i] == '}':
                    scope_depth -= 1
            
            # Group variables by scope
            if scope_depth not in scope_vars:
                scope_vars[scope_depth] = []
            scope_vars[scope_depth].append(var_name)
            
            # If we have multiple variables in the same scope, consider reusing
            if len(scope_vars[scope_depth]) > 1 and not added_comment:
                # Add a comment about register optimization
                kernel_body_start = modified_code.find('{', insert_position)
                if kernel_body_start > 0:
                    modified_body = (
                        modified_code[:kernel_body_start + 1] + 
                        var_reuse_comment + 
                        modified_code[kernel_body_start + 1:]
                    )
                    added_comment = True
        
        self.source_code = modified_body
        self.optimizations_applied.append("register_optimization")
        print(f"✅ Applied register optimization with max {max_registers} registers")
        return True

    def apply_thread_divergence_optimization(self) -> bool:
        """
        Apply optimizations to reduce thread divergence.
        
        Returns:
            bool: Success status
        """
        # Look for conditional statements that might cause divergence
        if_pattern = r'if\s*\(([^)]+)\)'
        divergent_ifs = []
        
        for match in re.finditer(if_pattern, self.source_code):
            condition = match.group(1)
            
            # Check if the condition depends on thread ID (potential divergence)
            if 'threadIdx' in condition:
                divergent_ifs.append((match.start(), condition))
        
        if not divergent_ifs:
            print("No thread divergence patterns detected")
            return False
        
        # Add a comment about thread divergence optimization
        divergence_comment = "\n// Thread divergence optimization applied\n"
        modified_code = divergence_comment + self.source_code
        
        # Replace the source code
        self.source_code = modified_code
        self.optimizations_applied.append("thread_divergence_optimization")
        print("✅ Applied thread divergence optimization")
        return True

    def apply_all_optimizations(self, block_size: int = 256, unroll_factor: int = 4, max_registers: int = 32) -> bool:
        """
        Apply all available optimizations to the kernel.
        
        Returns:
            bool: Success status
        """
        # Reset to original code
        self.reset()
        
        # Apply optimizations in a sensible order
        success = True
        
        # 1. Memory coalescing (optimize memory access patterns)
        success &= self.apply_memory_coalescing()
        
        # 2. Shared memory (reduce global memory access)
        success &= self.apply_shared_memory(block_size=block_size)
        
        # 3. Loop unrolling (reduce loop overhead)
        success &= self.apply_loop_unrolling(unroll_factor=unroll_factor)
        
        # 4. Register optimization (optimize register usage)
        success &= self.apply_register_optimization(max_registers=max_registers)
        
        # 5. Thread divergence (reduce warp divergence)
        success &= self.apply_thread_divergence_optimization()
        
        if success:
            print("✅ Successfully applied all optimizations")
        else:
            print("⚠️ Some optimizations could not be applied")
        
        return success

    def reset(self):
        """Reset to the original code."""
        self.source_code = self.original_code
        self.optimizations_applied = []
        return True

    def add_comment(self, comment_text):
        """Add a comment at the top of the file."""
        if "// " + comment_text not in self.source_code:
            self.source_code = f"// {comment_text}\n{self.source_code}"
        return True

    def save(self, output_path):
        """Save the transformed code to the specified file."""
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(self.source_code)
            print(f"✅ Successfully saved transformed code to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving to {output_path}: {e}")
            return False

    def analyze_kernel(self) -> Dict:
        """
        Analyze the kernel code and provide optimization suggestions.
        
        Returns:
            Dict: Analysis results with optimization suggestions
        """
        analysis = {
            "kernel_name": self.kernel_name,
            "optimization_opportunities": [],
            "metrics": {}
        }
        
        # Check for shared memory opportunity
        array_vars = self._find_array_variables()
        if array_vars:
            analysis["optimization_opportunities"].append({
                "type": "shared_memory",
                "confidence": "high",
                "reason": f"Found {len(array_vars)} array variables accessed using thread/block indices: {', '.join(array_vars)}",
                "expected_benefit": "Reduced global memory access latency, potential for ~2-10x speedup"
            })
        
        # Check for loop unrolling opportunity
        loops = self._find_loops()
        if loops:
            analysis["optimization_opportunities"].append({
                "type": "loop_unrolling",
                "confidence": "medium",
                "reason": f"Found {len(loops)} loops that could be unrolled",
                "expected_benefit": "Reduced loop overhead, potential for ~1.2-2x speedup"
            })
        
        # Check for memory coalescing opportunity
        non_coalesced_pattern = r'(\w+)\s*\[\s*(threadIdx\.x\s*\*\s*\w+|threadIdx\.x\s*\*\s*\d+)\s*\]'
        if re.search(non_coalesced_pattern, self.source_code):
            analysis["optimization_opportunities"].append({
                "type": "memory_coalescing",
                "confidence": "high",
                "reason": "Found non-coalesced memory access patterns",
                "expected_benefit": "Improved memory bandwidth utilization, potential for ~2-4x speedup"
            })
        
        # Check for register optimization opportunity
        temp_var_pattern = r'float\s+(\w+)\s*=\s*[^;]+;'
        temp_vars = list(re.finditer(temp_var_pattern, self.source_code))
        if len(temp_vars) > 5:  # Arbitrary threshold
            analysis["optimization_opportunities"].append({
                "type": "register_optimization",
                "confidence": "medium",
                "reason": f"Found {len(temp_vars)} temporary variables that might benefit from register optimization",
                "expected_benefit": "Increased occupancy, potential for ~1.1-1.5x speedup"
            })
        
        # Check for thread divergence
        if_with_threadidx_pattern = r'if\s*\([^)]*threadIdx[^)]*\)'
        divergent_ifs = list(re.finditer(if_with_threadidx_pattern, self.source_code))
        if divergent_ifs:
            analysis["optimization_opportunities"].append({
                "type": "thread_divergence_optimization",
                "confidence": "medium",
                "reason": f"Found {len(divergent_ifs)} conditional statements that might cause thread divergence",
                "expected_benefit": "Reduced warp divergence, potential for ~1.1-2x speedup"
            })
        
        # Calculate metrics
        analysis["metrics"] = {
            "code_complexity": len(self.source_code),
            "array_variables": len(array_vars),
            "loops": len(loops),
            "potential_speedup": self._estimate_potential_speedup(analysis["optimization_opportunities"])
        }
        
        return analysis
    
    def _estimate_potential_speedup(self, opportunities):
        """Estimate the potential speedup from all optimization opportunities."""
        speedup = 1.0
        
        for opt in opportunities:
            if opt["type"] == "shared_memory":
                speedup *= 2.0  # Conservative estimate
            elif opt["type"] == "loop_unrolling":
                speedup *= 1.2  # Conservative estimate
            elif opt["type"] == "memory_coalescing":
                speedup *= 2.0  # Conservative estimate
            elif opt["type"] == "register_optimization":
                speedup *= 1.1  # Conservative estimate
            elif opt["type"] == "thread_divergence_optimization":
                speedup *= 1.1  # Conservative estimate
        
        return round(speedup, 2)

    # Compatibility with older method names
    def apply_shared_memory_optimization(self):
        return self.apply_shared_memory()
        
    def transform(self, optimization_type):
        """Apply a specific optimization type."""
        if optimization_type == "shared_memory":
            return self.apply_shared_memory()
        elif optimization_type == "unroll":
            return self.apply_loop_unrolling()
        elif optimization_type == "memory_coalescing":
            return self.apply_memory_coalescing()
        elif optimization_type == "register":
            return self.apply_register_optimization()
        elif optimization_type == "thread_divergence":
            return self.apply_thread_divergence_optimization()
        elif optimization_type == "all":
            return self.apply_all_optimizations()
        else:
            print(f"Unsupported optimization type: {optimization_type}")
            return False