from pxr import Usd, UsdShade, Sdf, Gf

def apply_material_to_object():
    # Get the stage (USD stage) of the active scene
    stage = omni.usd.get_context().get_stage()

    # Load or create a material in the stage
    material = create_material(stage)

    # Assume the object is named "Cube" in the scene
    object_name = "Cube"

    # Get the prim (object) where the material will be applied
    prim = stage.GetPrimAtPath(f"/World/{object_name}")
    
    if not prim:
        print(f"Object '{object_name}' not found!")
        return
    
    # Create a shading node and connect it to the material
    material_binding_api = UsdShade.MaterialBindingAPI(prim)
    
    # Bind the material directly (ensure `material` is a UsdShade.Material object)
    material_binding_api.Bind(material)

    print(f"Material applied to {object_name}")

def create_material(stage):
    # Define the material's path in the USD stage
    material_path = "/World/Materials/MyMaterial"

    # Create a new material if it does not exist
    material_prim = stage.DefinePrim(material_path, "Xform")
    material = UsdShade.Material(material_prim)

    # Create a shader for the material (e.g., a lambert shader)
    shader_prim = stage.DefinePrim(f"{material_path}/LambertShader", "Shader")
    shader = UsdShade.Shader(shader_prim)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Set properties for the shader (e.g., base color)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))  # Red color

    # Connect the shader to the material
    material.CreateSurfaceOutput().ConnectToSource(shader, "surface")

    return material

if __name__ == "__main__":
    print("able to run script")