#NON-LOCAL ATTENTION MODULE
class NLAM:
  def __init__(self,x):
    g_x=Conv2D(32,1, padding='same')(x)
    teta_x=Conv2D(32,1, padding='same')(x)
    pi_x=Conv2D(32,1, padding='same')(x)
    f1=tf.keras.layers.Multiply()([teta_x,pi_x])
    f_div_c=tf.keras.activations.softmax(f1)
    f2=tf.keras.layers.Multiply()([f_div_c,g_x])
    self.z=f2+x
